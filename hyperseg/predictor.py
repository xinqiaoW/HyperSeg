# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from typing import Optional, Tuple, List

from .utils.transforms import ResizeLongestSide


class SamPredictor:
    def __init__(
        self,
        sam_model,
        wavelengths: List[float] = None,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model: The model to use for mask prediction.
          wavelengths: List of wavelengths for the hyperspectral image.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.rgb_encoder.img_size)
        self.wavelengths = wavelengths
        self.reset_image()

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        hq_token_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        images: torch.Tensor,  # B, H, W, C (note: H, W, C format)
        point_coords: Optional[torch.Tensor],  # B, N, 2
        point_labels: Optional[torch.Tensor],  # B, N
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = True,
        hq_token_only: bool = False,
        wavelengths: List[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          images (torch.Tensor): Input images with shape BxHxWxC.
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          wavelengths (List[float]): List of wavelengths for the hyperspectral image.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
          None: Placeholder for compatibility (previously returned all_features).
        """
        b, h, w, c = images.shape
        images = images.permute(0, 3, 1, 2)  # B, C, H, W

        if images.max() > 1:
            images = (images - images.min()) / (images.max() - images.min())

        # Use provided wavelengths or fall back to instance wavelengths
        wl = wavelengths if wavelengths is not None else self.wavelengths

        batched_input = []
        for i in range(point_coords.shape[0]):
            batched_input.append(
                {
                    "image": images[i] if images.shape[0] > 1 else images[0],
                    "original_size": (h, w),
                    "point_coords": point_coords[i].unsqueeze(0),
                    "point_labels": point_labels[i].unsqueeze(0),
                }
            )

        output = self.model(batched_input, wavelengths=wl, multimask_output=multimask_output)

        if return_logits:
            masks = torch.cat([out["logits"] for out in output], dim=0)
        else:
            masks = torch.cat([out["masks"] for out in output], dim=0)
        iou_predictions = torch.cat([out["iou_pred"] for out in output], dim=0)
        low_res_masks = torch.cat([out["low_res_logits"] for out in output], dim=0)

        return masks, iou_predictions, low_res_masks, None

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
