# import os
# import sys
# import numpy as np
# import tensorflow as tf
# import matplotlib
# import matplotlib.pyplot as plt
# import keras
#
# # Root directory of the project
# ROOT_DIR = os.path.abspath(r"C:\Users\tdewa\Code\snapsort_project\SnapSort")
#
# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
# from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# from mrcnn.model import log
#
#
# # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#
# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)
#
# # Path to Shapes trained weights
# SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")
#
# # MS COCO Dataset
# config = coco.CocoConfig()
#
# DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
#
# def get_ax(rows=1, cols=1, size=16):
#     """Return a Matplotlib Axes array to be used in all visualizations in the notebook."""
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax
#
# # Create model in inference mode
# with tf.device(DEVICE):
#     model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
#
# # Set weights file path
# if config.NAME == "shapes":
#     weights_path = SHAPES_MODEL_PATH
# elif config.NAME == "coco":
#     weights_path = COCO_MODEL_PATH
# # Or, uncomment to load the last model you trained
# # weights_path = model.find_last()
#
# # Load weights
# print("Loading weights ", weights_path)
# model.load_weights(weights_path, by_name=True)
#
# # Show stats of all trainable weights
# visualize.display_weight_stats(model)
#
# # Pick layer types to display
# LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']
# # Get layers
# layers = model.get_trainable_layers()
# layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES, layers))
# # Display Histograms
# fig, ax = plt.subplots(len(layers), 2, figsize=(10, 3*len(layers)),
#                        gridspec_kw={"hspace":1})
# for l, layer in enumerate(layers):
#     weights = layer.get_weights()
#     for w, weight in enumerate(weights):
#         tensor = layer.weights[w]
#         ax[l, w].set_title(tensor.name)
#         _ = ax[l, w].hist(weight[w].flatten(), 50)
#
# # Assuming `model` is your trained model
# model_path = os.path.join("path_to_save_weights", "mask_rcnn_custom.h5")
# model.keras_model.save_weights(model_path)


