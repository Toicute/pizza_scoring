'''
==================================================================================================
Because can't install detectron2 in local env so i need to move some Detectron source code to here
==================================================================================================
'''
import inspect
from contextlib import contextmanager
from functools import wraps
import math
from typing import List, Tuple, Optional

import numpy as np 
import torch
from torch import nn, device
from torch.nn import functional as F
from torchvision.ops import roi_align


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

try:
    from torch.fx._symbolic_trace import is_fx_tracing as is_fx_tracing_current

    tracing_current_exists = True
except ImportError:
    tracing_current_exists = False

try:
    from torch.fx._symbolic_trace import _orig_module_call

    tracing_legacy_exists = True
except ImportError:
    tracing_legacy_exists = False


@torch.jit.ignore
def is_fx_tracing_legacy() -> bool:
    """
    Returns a bool indicating whether torch.fx is currently symbolically tracing a module.
    Can be useful for gating module logic that is incompatible with symbolic tracing.
    """
    return torch.nn.Module.__call__ is not _orig_module_call


def is_fx_tracing() -> bool:
    """Returns whether execution is currently in
    Torch FX tracing mode"""
    if torch.jit.is_scripting():
        return False
    if TORCH_VERSION >= (1, 10) and tracing_current_exists:
        return is_fx_tracing_current()
    elif tracing_legacy_exists:
        return is_fx_tracing_legacy()
    else:
        # Can't find either current or legacy tracing indication code.
        # Enabling this assert_fx_safe() call regardless of tracing status.
        return False


def assert_fx_safe(condition: bool, message: str) -> torch.Tensor:
    """An FX-tracing safe version of assert.
    Avoids erroneous type assertion triggering when types are masked inside
    an fx.proxy.Proxy object during tracing.
    Args: condition - either a boolean expression or a string representing
    the condition to test. If this assert triggers an exception when tracing
    due to dynamic control flow, try encasing the expression in quotation
    marks and supplying it as a string."""
    # Must return a concrete tensor for compatibility with PyTorch <=1.8.
    # If <=1.8 compatibility is not needed, return type can be converted to None
    if torch.jit.is_scripting() or is_fx_tracing():
        return torch.zeros(1)
    return _do_assert_fx_safe(condition, message)


def _do_assert_fx_safe(condition: bool, message: str) -> torch.Tensor:
    try:
        if isinstance(condition, str):
            caller_frame = inspect.currentframe().f_back
            torch._assert(eval(condition, caller_frame.f_globals, caller_frame.f_locals), message)
            return torch.ones(1)
        else:
            torch._assert(condition, message)
            return torch.ones(1)
    except torch.fx.proxy.TraceError as e:
        print(
            "Found a non-FX compatible assertion. Skipping the check. Failure is shown below"
            + str(e)
        )


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
    

@torch.jit.script_if_tracing
def paste_masks_in_image(
    masks: torch.Tensor, boxes: torch.Tensor, image_shape: Tuple[int, int], threshold: float = 0.5
):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

        from torchvision import __version__

        version = tuple(int(x) for x in __version__.split(".")[:2])
        # https://github.com/pytorch/vision/pull/2438
        assert version >= (0, 7), "Require torchvision >= 0.7"

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        if input.is_quantized:
            input = input.dequantize()
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
    

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            [isinstance(t, torch.Tensor) for t in x]
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)
    

def assign_boxes_to_levels(
    box_lists: List[Boxes],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


# script the module to avoid hardcoded device type
@torch.jit.script_if_tracing
def _convert_boxes_to_pooler_format(boxes: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    sizes = sizes.to(device=boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(sizes), dtype=boxes.dtype, device=boxes.device), sizes
    )
    return cat([indices[:, None], boxes], dim=1)


def convert_boxes_to_pooler_format(box_lists: List[Boxes]):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """
    boxes = torch.cat([x.tensor for x in box_lists], dim=0)
    # __len__ returns Tensor in tracing.
    sizes = shapes_to_tensor([x.__len__() for x in box_lists])
    return _convert_boxes_to_pooler_format(boxes, sizes)


@torch.jit.script_if_tracing
def _create_zeros(
    batch_target: Optional[torch.Tensor],
    channels: int,
    height: int,
    width: int,
    like_tensor: torch.Tensor,
) -> torch.Tensor:
    batches = batch_target.shape[0] if batch_target is not None else 0
    sizes = (batches, channels, height, width)
    return torch.zeros(sizes, dtype=like_tensor.dtype, device=like_tensor.device)
    

class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as 1/s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        if not is_fx_tracing():
            torch._assert(
                isinstance(x, list) and isinstance(box_lists, list),
                "Arguments to pooler must be lists",
            )
        assert_fx_safe(
            len(x) == num_level_assignments,
            "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
                num_level_assignments, len(x)
            ),
        )
        assert_fx_safe(
            len(box_lists) == x[0].size(0),
            "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
                x[0].size(0), len(box_lists)
            ),
        )
        if len(box_lists) == 0:
            return _create_zeros(None, x[0].shape[1], *self.output_size, x[0])

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])

        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))

        return output