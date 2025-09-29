import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T

import torch
import torch.nn.functional as F
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.layers import cat
from detectron2.structures import Instances
from detectron2.modeling.box_regression import smooth_l1_loss
from detectron2.modeling import build_model
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY

# ---- Step 1: Register the COCO-style dataset ----
DATASET_ROOT = "../datasets/600"

register_coco_instances(
    "my_dataset_train", {},
    os.path.join(DATASET_ROOT, "annotations/instances_train_fixed.json"),
    os.path.join(DATASET_ROOT, "train")
)

register_coco_instances(
    "my_dataset_val", {},
    os.path.join(DATASET_ROOT, "annotations/instances_val_fixed.json"),
    os.path.join(DATASET_ROOT, "val")
)


# ---- Step 2: Define a custom Trainer with Evaluator support ----
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(cfg, is_train=True)
        )
        
def iou_loss(pred, target, epsilon=1e-6):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    iou = intersection / (union + epsilon)
    return 1 - iou


# ---- Step A: Custom Output Layer (Remove weighted classification loss) ----
class WeightedFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions
        gt_classes = cat([p.gt_classes for p in proposals], dim=0).to(scores.device)

        # Classification loss without class weights
        loss_cls = F.cross_entropy(scores, gt_classes)

        # Box regression loss (standard)
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        gt_boxes = cat([p.gt_boxes.tensor for p in proposals], dim=0)
        regression_targets = self.box2box_transform.get_deltas(proposal_boxes, gt_boxes)

        fg_inds = torch.nonzero(gt_classes != self.num_classes, as_tuple=True)[0]
        if len(fg_inds) > 0:
            loss_box_reg = smooth_l1_loss(
                proposal_deltas[fg_inds],
                regression_targets[fg_inds],
                beta=1.0,
                reduction="sum",
            ) / gt_classes.numel()
        else:
            loss_box_reg = proposal_deltas.sum() * 0

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}


# ---- Step B: Custom ROI Heads with Weighted Mask Loss ----
@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
    def _init_box_head(self, cfg):
        super()._init_box_head(cfg)
        self.box_predictor = WeightedFastRCNNOutputLayers(cfg, self.box_head.output_shape)
        
    

    def mask_rcnn_loss(self, pred_mask_logits, instances):
        """
        Override mask loss to use IoU loss.
        """
        if len(instances) == 0:
            return {"loss_mask": pred_mask_logits.sum() * 0}

        cls_weights = torch.tensor([1.0, 1.5, 10], device=pred_mask_logits.device)

        gt_classes = []
        gt_masks = []
        pred_masks = []

        for i, inst in enumerate(instances):
            if len(inst) == 0:
                continue

            gt_masks.append(inst.gt_masks.crop_and_resize(
                inst.pred_boxes.tensor, pred_mask_logits.shape[-2:]
            ))
            gt_classes.append(inst.gt_classes)
            pred_masks.append(pred_mask_logits[i, inst.gt_classes])

        gt_masks = torch.cat(gt_masks, dim=0).to(dtype=torch.float32)
        gt_classes = torch.cat(gt_classes, dim=0)
        pred_masks = torch.cat(pred_masks, dim=0)

        # Apply per-class weights
        weights = cls_weights[gt_classes].view(-1, 1, 1)

        # IoU loss
        loss = iou_loss(pred_masks, gt_masks)

        return {"loss_mask": loss}




# ---- Step C: Update Trainer to use Custom ROI Heads ----
class MyCustomTrainer(MyTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        # Ensure model uses CustomROIHeads (already registered)
        print("[INFO] Using custom ROI heads with class-weighted mask loss")
        return model


# ---- Step 3: Set up the configuration ----
cfg = get_cfg()

# Load base model config
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)

# Set pretrained weights # Replace with checkpoint when retraining
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

# Datasets
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

# IMPORTANT: Set number of classes to match your dataset (excluding background)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Update this based on your classes

# Training hyperparameters
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []  # Avoid learning rate decay warnings

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.TEST.EVAL_PERIOD = 200  # Evaluates every 200 iterations

# cfg.MODEL.ROI_HEADS.POOLER_RESOLUTION = 28  # Increase resolution to improve mask accuracy

cfg.OUTPUT_DIR = "./output_custom"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Optional: Save config for reproducibility
with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())

# ---- Step 4: Train ----
trainer = MyCustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ---- Step 5: Evaluate After Training ----
evaluator = COCOEvaluator(dataset_name="my_dataset_val", tasks=("bbox", "segm"), output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
