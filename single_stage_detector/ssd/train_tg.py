from extra.models import retinanet
from extra.models.resnet import ResNeXt50_32X4D
import pdb
from tinygrad.nn.state import get_parameters, get_state_dict
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

if __name__ == "__main__":
	backbone = ResNeXt50_32X4D()
	
	model = retinanet.RetinaNet(backbone)

	params = get_parameters(model)
	
	breakpoint()