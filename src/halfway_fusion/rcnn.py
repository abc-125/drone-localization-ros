from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

from typing import Dict, List, Optional
import torch

           
@META_ARCH_REGISTRY.register()
class HalfwayFusionRCNN(GeneralizedRCNN):

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image_rgb: Tensor, rgb image in (C, H, W) format.
                * image_t: Tensor, thermal image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images_rgb, images_t = self.preprocess_image(batched_inputs)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # get features from both types of images:
        features = self.backbone(images_rgb.tensor, images_t.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images_rgb, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images_rgb, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

 
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images_rgb, images_t = self.preprocess_image(batched_inputs)

        # get features from both types of images:
        features = self.backbone(images_rgb.tensor, images_t.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images_rgb, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images_rgb, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images_rgb.image_sizes)
        else:
            return results

           
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch both types of the input images.
        """
        images_rgb = [x["image_rgb"].to(self.device) for x in batched_inputs]
        images_rgb = [(im - self.pixel_mean) / self.pixel_std for im in images_rgb]
        images_rgb = ImageList.from_tensors(images_rgb, self.backbone.size_divisibility)
        
        images_t = [x["image_t"].to(self.device) for x in batched_inputs]
        images_t = [(im - self.pixel_mean) / self.pixel_std for im in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)
        
        return images_rgb, images_t
