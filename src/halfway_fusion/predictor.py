from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog

import torch


class HalfwayFusionPredictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        
    def __call__(self, images):
        original_image_rgb, original_image_t = images[0], images[1]

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image_rgb = original_image_rgb[:, :, ::-1]
                original_image_t = original_image_t[:, :, ::-1]
                
            height, width = original_image_rgb.shape[:2]

            image_rgb = torch.as_tensor(original_image_rgb.astype("float32").transpose(2, 0, 1))
            image_t = torch.as_tensor(original_image_t.astype("float32").transpose(2, 0, 1))

            inputs = {"image_rgb": image_rgb, "image_t": image_t, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            
            return predictions