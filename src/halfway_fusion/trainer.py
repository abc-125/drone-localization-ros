from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

import copy
import torch
import os


def halfway_fusion_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict) 
    
    # read rgb and thermal images:
    image_rgb = utils.read_image(dataset_dict["file_name_rgb"], format="RGB")
    image_t = utils.read_image(dataset_dict["file_name_t"], format="L")
    
    image_rgb = torch.from_numpy(image_rgb.transpose(2, 0, 1).astype("float32"))
    image_t = torch.from_numpy(image_t.transpose(2, 0, 1).astype("float32"))

    annos = [annotation for annotation in dataset_dict.pop("annotations")]

    return {
       # create the format that the model expects
       "image_rgb": image_rgb,
       "image_t": image_t,
       "instances": utils.annotations_to_instances(annos, image_rgb.shape[1:]),
       "height": dataset_dict["height"],
       "width": dataset_dict["width"],
       "image_id": dataset_dict["image_id"]
    }


class HalfwayFusionTrainer(DefaultTrainer):
    '''Rewrites build_detection_test_loader and build_train_loader with my mapper. Uses COCOEvaluator while training.'''

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=halfway_fusion_mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=halfway_fusion_mapper)