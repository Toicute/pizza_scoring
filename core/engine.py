import torch
import yaml
from typing import Tuple, Dict, Any

from torchvision import transforms
import numpy as np

from .utils import non_max_suppression_mask_conf, letterbox
from .tmp_detectron import paste_masks_in_image, retry_if_cuda_oom, ROIPooler, Boxes


class Yolov7Segmentation:
    def __init__(self, model_weight: str, model_config: str) -> None:
        """
        Initialize the Yolov7Segmentation model.

        Args:
            model_weight (str): Path to the model weights.
            model_config (str): Path to the model configuration.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hyp = self.load_hyperparameters(model_config)
        self.model = self.load_model(model_weight)
        self.model.half().to(self.device).eval()

    def load_hyperparameters(self, config_path: str) -> Dict[str, Any]:
        """
        Load hyperparameters from a YAML configuration file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Dict[str, Any]: Hyperparameters.
        """
        with open(config_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def load_model(self, weight_path: str) -> torch.nn.Module:
        """
        Load the model from a weight file.

        Args:
            weight_path (str): Path to the weight file.

        Returns:
            torch.nn.Module: Loaded model.
        """
        weights = torch.load(weight_path)
        return weights['model']

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the input image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = letterbox(image.copy(), 640, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        return image.to(self.device).half()

    def postprocess_output(self, output: Dict[str, Any], image_shape: Tuple[int, int], conf_thres: float, iou_thres: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Postprocess the model output.

        Args:
            output (Dict[str, Any]): Model output.
            image_shape (Tuple[int, int]): Shape of the input image.
            conf_thres (float): Confidence threshold.
            iou_thres (float): IoU threshold.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed predictions and masks.
        """
        inf_out, attn, bases, sem_output = output['test'], output['attn'], output['bases'], output['sem']
        bases = torch.cat([bases, sem_output], dim=1)

        pooler = ROIPooler(
            output_size=self.hyp['mask_resolution'],
            scales=(self.model.pooler_scale,),
            sampling_ratio=1,
            pooler_type='ROIAlignV2',
            canonical_level=2
        )

        output, output_mask, *_ = non_max_suppression_mask_conf(
            inf_out, attn, bases, pooler, self.hyp,
            conf_thres=conf_thres, iou_thres=iou_thres, merge=False, mask_iou=None
        )

        pred, pred_masks = output[0], output_mask[0]
        original_pred_masks = pred_masks.view(-1, self.hyp['mask_resolution'], self.hyp['mask_resolution'])
        bboxes = Boxes(pred[:, :4])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            original_pred_masks, bboxes, image_shape, threshold=0.5
        )

        return pred, pred_masks

    def inference(self, image: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.65) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform inference on the input image.

        Args:
            image (np.ndarray): Input image.
            conf_thres (float): Confidence threshold.
            iou_thres (float): IoU threshold.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predicted masks, classes, and confidences.
        """
        preprocessed_image = self.preprocess_image(image)
        output = self.model(preprocessed_image)
        pred, pred_masks = self.postprocess_output(output, preprocessed_image.shape[-2:], conf_thres, iou_thres)

        pred_masks_np = pred_masks.detach().cpu().numpy()
        pred_cls = pred[:, 5].detach().cpu().numpy()
        pred_conf = pred[:, 4].detach().cpu().numpy()
        
        return pred_masks_np, pred_cls, pred_conf