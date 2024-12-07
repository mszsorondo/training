from tinygrad.tensor import Tensor
from tinygrad import dtypes
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute the Intersection over Union (IoU) for two sets of bounding boxes.

    :param boxes1: Tensor of shape [N, 4]
    :param boxes2: Tensor of shape [M, 4]
    :return: Tensor of shape [N, M] with IoU values
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou

def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> (Tensor, Tensor):
    """
    Compute the intersection and union areas for two sets of bounding boxes.

    :param boxes1: Tensor of shape [N, 4]
    :param boxes2: Tensor of shape [M, 4]
    :return: Tensors of shape [N, M] for intersection and union areas
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Compute the top-left and bottom-right corners of the intersection
    lt = Tensor.maximum(boxes1[:, None, :2], boxes2[:, :2])  # Shape: [N, M, 2]
    rb = Tensor.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # Shape: [N, M, 2]

    # Compute width and height of the intersection
    wh = _upcast(rb - lt)
    wh = Tensor.maximum(wh, Tensor(0))  # Clip values below 0

    inter = wh[:, :, 0] * wh[:, :, 1]  # Intersection area, shape: [N, M]

    # Compute union area
    union = area1[:, None] + area2 - inter  # Shape: [N, M]
    return inter, union

def box_area(boxes: Tensor) -> Tensor:
    """
    Compute the area of bounding boxes.

    :param boxes: Tensor of shape [N, 4] where each row is [x1, y1, x2, y2]
    :return: Tensor of shape [N] with the area of each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def _upcast(t: Tensor) -> Tensor:
    """
    Protects from numerical overflows in multiplications by upcasting to a higher equivalent type.
    """
    if t.dtype in (dtypes.float32, dtypes.float64):
        return t
    elif t.dtype in (dtypes.float16,):
        return t.cast(dtypes.float32)
    elif t.dtype in (dtypes.int32, dtypes.int64):
        return t
    else:
        return t.cast(dtypes.int32)

def _upcast(t: Tensor) -> Tensor:
    """
    def _upcast(self,t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if np.issubdtype(t.dtype, np.floating):
        return t if t.dtype in (np.float32, np.float64) else t.astype(np.float32)
    else:
        return t if t.dtype in (np.int32, np.int64) else t.astype(np.int32)
    """
    if t.dtype in (dtypes.float32, dtypes.float64):
        return t  # Already at a safe floating-point type
    elif t.dtype in (dtypes.float16,):  # Check for lower-precision floats
        return t.cast(dtypes.float32)
    elif t.dtype in (dtypes.int32, dtypes.int64):
        return t  # Already at a safe integer type
    else:  # Upcast all other integer types
        return t.cast(dtypes.int32)
