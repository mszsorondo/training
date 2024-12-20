from extra.models import retinanet
from extra.models.resnet import ResNeXt50_32X4D
import pdb
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad import Tensor, dtypes, Device
from typing import List, Tuple, Dict, Optional
from torch import save as torchsave
from torch import load as torchload
import sys, os

#tinygrad training
"""
pytorch parameters
backbone='resnext50_32x4d'
trainable_backbone_layers=3, 
sync_bn=False, 
data_layout='channels_last', 
amp=True, 
dataset='openimages-mlperf', 
data_path='/datasets/open-images-v6', 
image_size=[800, 800], 
data_augmentation='hflip', 
epochs=26, 
start_epoch=0, output_dir=None, target_map=0.34, resume='', pretrained=False, batch_size=2, eval_batch_size=2, 
lr=0.0001, warmup_epochs=1, warmup_factor=0.001, workers=4, print_freq=20, eval_print_freq=20, 
test_only=False, seed=4044681145, device='cuda', world_size=1, dist_url='env://')

"""
class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Args:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: Device) -> 'ImageList':
        print("WARNING: This is an exact copy of the MLPERF class, method not tested")
        raise NotImplementedError
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class TrainingRegister:
    def __init__(self,model):
        self.model = model
        self.forward_original_image_sizes = None

    def store_original_image_sizes(self, images):
        self.original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            self.original_image_sizes.append((val[0], val[1]))

def frozen_bn_forward2(self,x:Tensor):

            batch_mean = self.running_mean
            # NOTE: this can be precomputed for static inference. we expand it here so it fuses
            batch_invstd = self.running_var.reshape(1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()
            return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd)
def frozen_bn_forward(self, x):
    scale = self.weight * self.running_var.rsqrt()
    bias = self.bias - self.running_mean * scale
    scale = scale.reshape(1, -1, 1, 1)
    bias = bias.reshape(1, -1, 1, 1)
    return x * scale + bias

def frozen_bn_forward_torchvision(self,x):
    w = self.weight.reshape(1, -1, 1, 1)
    b = self.bias.reshape(1, -1, 1, 1)
    rv = self.running_var.reshape(1, -1, 1, 1)
    rm = self.running_mean.reshape(1, -1, 1, 1)
    scale = w * (rv + self.eps).rsqrt()
    bias = b - rm * scale
    return x * scale + bias

class AnchorGenerator:
    def __init__(self):
        self.sizes = ((32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812))
        self.aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.sizes)
        self.cell_anchors = [self.generate_anchors(size, aspect_ratio)
                             for size, aspect_ratio in zip(self.sizes, self.aspect_ratios)]
    def generate_anchors(self, scales: List[int], aspect_ratios: List[float], dtype = dtypes.float32,
                         device = Device.DEFAULT):
        from tinygrad import Tensor as tgTensor
        scales = tgTensor(scales, dtype=dtype, device=device)
        aspect_ratios = tgTensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = aspect_ratios.sqrt()
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = (-ws).stack(-hs, ws, hs, dim=1) / 2
        return base_anchors.round()

    def __call__(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        return self.forward(image_list, feature_maps)

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[Tensor(image_size[0] // g[0], dtype=dtypes.int64, device=device),
                    Tensor(image_size[1] // g[1], dtype=dtypes.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        

        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [anchors_per_image[0].cat(*anchors_per_image[1:]) for anchors_per_image in anchors]
        return anchors
    def set_cell_anchors(self, dtype: dtypes, device):
        self.cell_anchors = [cell_anchor.to(device=device) for cell_anchor in self.cell_anchors]

    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[float]]) -> List[Tensor]:
        """
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # Verificación de tamaños consistentes
        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError("Anchors should be Tuple[Tuple[int]] because each feature "
                             "map could potentially have different sizes and aspect ratios. "
                             "There needs to be a match between the number of "
                             "feature maps passed and the number of sizes / aspect ratios specified.")

        # Iterar sobre cada nivel de mapa de características
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            # Crear los desplazamientos en X e Y
            shifts_x = Tensor.arange(0, grid_width).cast(dtypes.float32) * stride_width
            shifts_y = Tensor.arange(0, grid_height).cast(dtypes.float32) * stride_height
            # Crear la malla 2D (equivalente a meshgrid)
            shift_y, shift_x = shifts_y.expand(grid_width, -1).T, shifts_x.expand(grid_height, -1)

            # Aplanar los desplazamientos
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()

            # Combinar los desplazamientos
            shifts = shift_x.stack(shift_y, shift_x, shift_y, dim=1)

            # Ajustar los anchors base por los desplazamientos

            level_anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)

            # Añadir los anchors calculados para este nivel
            anchors.append(level_anchors)

        return anchors

class RetinaNetTrainer:
    def __init__(self, tg_model, reference_model, weights_from_disk=False, store_to_disk=False, anchor_generator=None):
        self.tg_model=tg_model
        self.reference_model=reference_model
        if store_to_disk:
            torchsave(self.reference_model.state_dict(), 'ref_model.pth')

        if weights_from_disk:
            self.reference_model.load_state_dict(torchload('ref_model.pth', weights_only=True))

        self.anchor_generator = AnchorGenerator() if anchor_generator is None else anchor_generator
        self.tg_model.set_training_anchor_generator(self.anchor_generator)
    def copy_params(self):
        from tinygrad.helpers import get_child 
        from torch import nn
        params = [p for p in self.reference_model.named_parameters()]
        tgdict = get_state_dict(self.tg_model)

        """this maybe also works for setting **some** params, try it again later
            state_dict = self.reference_model.state_dict()
                                for k, v in state_dict.items():
                                  obj = get_child(self.tg_model, k)
                                  dat = v.detach().numpy()
                                  assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
                                  obj.assign(dat)"""

        for name,param in params:
            tgdict[name].assign(param.clone().detach().numpy())
            tgdict[name].requires_grad=param.requires_grad
            #print(name)

        for (k,v) in self.reference_model.state_dict().items():
            if k in tgdict and k not in params:
                tgdict[k] = Tensor(v.numpy())
        
        self.tg_model.backbone.body.bn1.weight.assign(Tensor(self.reference_model.backbone.body.bn1.weight.detach().numpy()))
        self.tg_model.backbone.body.bn1.bias.assign(Tensor(self.reference_model.backbone.body.bn1.bias.detach().numpy()))
        self.tg_model.backbone.body.bn1.running_var.assign(Tensor(self.reference_model.backbone.body.bn1.running_var.detach().numpy()))
        self.tg_model.backbone.body.bn1.running_mean.assign(Tensor(self.reference_model.backbone.body.bn1.running_mean.detach().numpy()))

        self.tg_model.backbone.body.bn1.weight.requires_grad = self.reference_model.backbone.body.bn1.weight.requires_grad
        self.tg_model.backbone.body.bn1.bias.requires_grad = self.reference_model.backbone.body.bn1.bias.requires_grad
        self.tg_model.backbone.body.bn1.__class__.__call__ = frozen_bn_forward_torchvision
        self.tg_model.backbone.body.bn1.__call__ = frozen_bn_forward
        #batch norm weights and biases are not copied for some reason (models are not modules)
        for name, module in self.reference_model.named_modules():
            if module.__class__.__name__=="FrozenBatchNorm2d":
                get_state_dict(self.tg_model)[name + '.weight'].assign(Tensor(module.weight.detach().numpy()))
                get_state_dict(self.tg_model)[name + '.bias'].assign(Tensor(module.bias.detach().numpy()))
                get_state_dict(self.tg_model)[name + '.running_var'].assign(Tensor(module.running_var.detach().numpy()))
                get_state_dict(self.tg_model)[name + '.running_mean'].assign(Tensor(module.running_mean.detach().numpy()))
                get_state_dict(self.tg_model)[name + '.weight'].requires_grad = module.weight.requires_grad
                get_state_dict(self.tg_model)[name + '.bias'].requires_grad = module.bias.requires_grad

                





    def set_optimizer(self, lr):
        from tinygrad.nn.optim import Adam as Adam_tg
        params = [p for p in self.reference_model.parameters() if p.requires_grad]
        tg_params = [i for i in get_parameters(self.tg_model) if i.requires_grad]
    
        
        assert(len(params)==len(tg_params))
        print("Warning: should test that params are the same")
        #in torch:     optimizer = torch.optim.Adam(params, lr=args.lr)
        return Adam_tg([i for i in get_parameters(self.tg_model) if i.requires_grad], lr=lr)



def _resize_image_and_masks(image: Tensor,
                            target: Optional[Dict[str, Tensor]] = None,
                            image_size: Optional[Tuple[int, int]] = None,
                            ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    im_shape = Tensor(image.shape[-2:])

    image = image[None].interpolate(size=image_size, align_corners=False)[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = mask[:, None].float().interpolate(size=image_size)[:, 0].cast(dtypes.uint8)
        target["masks"] = mask
    return image, target


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    def stack_dim1(tensors):
        return Tensor([list(t) for t in zip(*[tensor.numpy() for tensor in tensors])])
    ratios = [
        Tensor(s, dtype=dtypes.float32, device=boxes.device) /
        Tensor(s_orig, dtype=dtypes.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = [boxes[:, i] for i in range(boxes.shape[1])] #unbind?

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return xmin.stack(ymin, xmax, ymax, dim=1) #if it doesnt work try stack_dim1
def resize_keypoints(keypoints: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    
    ratios = [
        Tensor(s, dtype=dtypes.float32, device=keypoints.device) /
        Tensor(s_orig, dtype=dtypes.float32, device=keypoints.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    resized_data[..., 0] *= ratio_w
    resized_data[..., 1] *= ratio_h
    
    return resized_data

class GeneralizedRCNNTransform:
    def __init__(self, image_size: Optional[Tuple[int, int]]= None,
                 image_mean: List[float] = None, image_std: List[float]= None,):
        self.image_size = image_size if image_size is not None else (800,800)
        self.image_mean = [0.485, 0.456, 0.406] if image_mean is None else image_mean
        self.image_std = [0.229, 0.224, 0.225] if image_std is None else image_std

    def normalize(self, image: Tensor) -> Tensor:
            if not image.is_floating_point():
                raise TypeError(
                    f"Expected input images to be of floating type (in range [0, 1]), "
                    f"but found type {image.dtype} instead"
                )
            dtype, device = image.dtype, image.device
            mean = Tensor(self.image_mean, dtype=dtype, device=device)
            std = Tensor(self.image_std, dtype=dtype, device=device)
            return (image - mean[:, None, None]) / std[:, None, None]

    def forward(self, images, targets):
        assert(targets is not None)
        targets_copy: List[Dict[str, Tensor]] = []
        for t in targets:
            data: Dict[str, Tensor] = {}
            for k, v in t.items():
                data[k] = v
            targets_copy.append(data)
        targets = targets_copy
        
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i]
            if image.ndim != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        
        images = images[0].stack(*images[1:], dim=0) #images = torch.stack(images)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
        #breakpoint()
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def resize(self,
               image: Tensor,
               target: Optional[Dict[str, Tensor]] = None,
               ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        image, target = _resize_image_and_masks(image, target, self.image_size)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target


if __name__ == "__main__":
    backbone = ResNeXt50_32X4D()
    
    model = retinanet.RetinaNet(backbone)

    params = get_parameters(model)
    
