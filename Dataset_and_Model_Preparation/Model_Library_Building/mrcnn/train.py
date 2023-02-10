"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import json
import skimage
import datetime
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


from nets.config import Config
from nets import model as modellib, utils
class CAPTCHAConfig(Config):
    """Configuration for training on the toy CAPTCHA dataset.
    Derives from the base Config class and overrides values specific
    to the toy CAPTCHA dataset.
    """
    # Give the configuration a recognizable name
    NAME = "CAPTCHA"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 10

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3651  # background + size of character set

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small


    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 11656

    # use small validation steps since the epoch is small

    command = 'train'

    weights = ''

    dataset = './char_dajie'

    logs = './logs'

    BACKBONE = 'resnet50'

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.
    }



class CAPTCHADataset(utils.Dataset):
    """Generates the CAPTCHA synthetic dataset. The dataset consists of simple
    CAPTCHA (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_CAPTCHA(self, dataset_dir, subset):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        with open('class_to_idx_dajie.txt', 'r', encoding='utf-8') as file:
            class_ids = file.readlines()
            n = 1
            for id in class_ids:
                temp = id.split(' ')
                char_label = temp[1].replace('\n', '')
                char_label = str(int(char_label) + 1)
                self.add_class("CAPTCHA", n, char_label)
                n += 1

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:

            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]

            num_ids = [int(n['label']) for n in objects]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "CAPTCHA",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a number dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "CAPTCHA":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "CAPTCHA":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """

        image = skimage.io.imread(self.image_info[image_id]['path'], plugin='pil')

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

def train(model):
    """Train the nets."""
    # Training dataset.
    dataset_train = CAPTCHADataset()
    dataset_train.load_CAPTCHA(config.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CAPTCHADataset()
    dataset_val.load_CAPTCHA(config.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='all')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run nets detection and generate the color splash effect
        print("Running on {}".format(config.image))
        # Read image
        image = skimage.io.imread(config.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print(r)
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    config = CAPTCHAConfig()
    if config.command == "train":
        print("--------train--------")
    else:
        class InferenceConfig(CAPTCHAConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            command = 'splash'

            weights = ''

            dataset = './dataset'

            logs = './logs'

            image = './497.jpg'

            NAME = "CAPTCHA"
        config = InferenceConfig()
    config.display()
    # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Train Mask R-CNN to detect CAPs.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'splash'")
    # parser.add_argument('--dataset', required=False,
    #                     metavar="/path/to/CAP/dataset/",
    #                     help='Directory of the CAP dataset')
    # parser.add_argument('--weights', required=True,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--image', required=False,
    #                     metavar="path or URL to image",
    #                     help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    # args = parser.parse_args()

    # Validate arguments
    if config.command == "train":
        assert config.dataset, "Argument --dataset is required for training"
    elif config.command == "splash":
        # assert config.image or config.video,\
        #        "Provide --image or --video to apply color splash"
        print("??????")

    print("Weights: ", config.weights)
    print("Dataset: ", config.dataset)
    print("Logs: ", config.logs)
    print("no pre trainec nets!!!!!!!!!!!!!!!!!!!!!!s1gh")
    # Configurations


    # Create nets
    if config.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=config.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=config.logs)
        model.load_weights(config.weights, by_name=True)

    # # Select weights file to load
    # if config.weights.lower() == "coco":
    #     weights_path = COCO_WEIGHTS_PATH
    #     # Download weights file
    #     if not os.path.exists(weights_path):
    #         utils.download_trained_weights(weights_path)
    # elif config.weights.lower() == "last":
    #     # Find last trained weights
    #     weights_path = nets.find_last()
    # elif config.weights.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     weights_path = nets.get_imagenet_weights()
    # else:
    #     weights_path = config.weights

    # # Load weights
    # print("Loading weights ", weights_path)
    # if config.weights.lower() == "coco":
    #     # Exclude the last layers because they require a matching
    #     # number of classes
    #     nets.load_weights(weights_path, by_name=True, exclude=[
    #         "mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask"])
    # else:
    #     if config.command == "train":
    #         nets.load_weights(weights_path, by_name=True, exclude=[
    #             "mrcnn_class_logits", "mrcnn_bbox_fc",
    #             "mrcnn_bbox", "mrcnn_mask"])
    #     elif config.command == "splash":
    #         nets.load_weights(weights_path, by_name=True)


    # Train or evaluate
    if config.command == "train":
        train(model)
    elif config.command == "splash":
        print("?????")
        detect_and_color_splash(model, image_path=config.image,
                                video_path=config.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(config.command))
