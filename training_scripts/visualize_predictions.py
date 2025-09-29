import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Load config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./output_custom/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("my_dataset_val",)  # Use this to get metadata
cfg.MODEL.DEVICE = "cuda"  # or "cpu"

predictor = DefaultPredictor(cfg)

# Path to test images
IMAGE_DIR = "../datasets/300/val"

# Output directory
os.makedirs("vis_results", exist_ok=True)

# Metadata
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

# Run inference and save results
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    save_path = os.path.join("vis_results", img_name)
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

print("Predictions saved in vis_results/")
