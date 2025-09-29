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

# ---- Step 1: Register the COCO-style dataset ----
DATASET_ROOT = "../datasets/600_2"

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
            # mapper=DatasetMapper(cfg, is_train=True, augmentations=augmentation)
        )

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

# Set pretrained weights using pkl model file
# cfg.MODEL.WEIGHTS = "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_200.pkl"

# Datasets
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

# IMPORTANT: Set number of classes to match your dataset (excluding background)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # ← UPDATE THIS if you fix the categories JSON

# Training hyperparameters
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = []  # Avoid learning rate decay warnings

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.TEST.EVAL_PERIOD = 200  # Evaluates every 200 iterations

cfg.OUTPUT_DIR = "./output_custom"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Optional: Save config for reproducibility
with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())

# ---- Step 4: Train ----
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ---- Step 5: Evaluate After Training ----
evaluator = COCOEvaluator(dataset_name="my_dataset_val", tasks=("bbox","segm"), output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))



