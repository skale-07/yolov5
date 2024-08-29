import cv2
import numpy as np
import matplotlib.pyplot as plt
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config

class InferenceConfig(Config):
    NAME = "coco_inference"
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + background
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir="path_to_logs", config=config)

# Load weights
model.load_weights('path_to_mask_rcnn_coco.h5', by_name=True)

# Load image
image = cv2.imread(r'C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\object_detection\sample_images\8512296263_5fc5458e20_z.jpg')
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            ['BG'] + list(range(1, 81)), r['scores'])
