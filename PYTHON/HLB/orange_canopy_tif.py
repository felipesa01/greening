# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:57:00 2021
@author: Felipe Rafael de Sá Menezes Lucena
"""

import os
import sys
import json
import numpy as np
import time
import datetime
import skimage.draw
from PIL import Image, ImageDraw

from imgaug import augmenters as iaa

ROOT_DIR = '../Mask_RCNN'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist.'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
import gdalbasics as gdb

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class OrangeCanopyConfig(Config):
    """Configuration for training on the orange's tree canopy dataset.
    Derives from the base Config class and overrides values specific
    to the orange's tree canopy dataset.
    """
    # Give the configuration a recognizable name
    NAME = "oranges_trees_canopy"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (trees canopy)

    # All of our training images are 512x512
    # IMAGE_MIN_DIM = 256       # Configurado abaixo
    # IMAGE_MAX_DIM = 256       # Configurado abaixo

    # You can experiment with this number to see if it improves training
    # Antes: 500
    # Valor baseado em gaciaBraga (trees.py): 4171
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    # Antes: 5
    # Valor baseado em gaciaBraga (trees.py): 1001
    VALIDATION_STEPS = 100

    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'  # garciaBraga (trees.py) tbm usa resnet50

    # To be honest, I haven't taken the time to figure out what these do
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)      # Configurado abaixo
    # TRAIN_ROIS_PER_IMAGE = 32     # Configurado abaixo
    # MAX_GT_INSTANCES = 50     # Configurado abaixo
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000

    # CigButts configura apenas os parâmetros acima. A partir desse ponto, são configurações beseadas em garciaBraga (trees.py)

    DETECTION_MIN_CONFIDENCE = 0.5
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    USE_MINI_MASK = False
    IMAGE_CHANNEL_COUNT = 3
    LEARNING_RATE = 0.0002
    LEARNING_MOMENTUM = 0.9
    # I left this equal to 220, but during the training my images had a maximum of 150 tree crowns (garciaBraga -> trees.py)
    # I left this equal to 100
    MAX_GT_INSTANCES = 100
    DETECTION_MAX_INSTANCES = 100
    TRAIN_ROIS_PER_IMAGE = 100
    RPN_TRAIN_ANCHORS_PER_IMAGE = 100
    #
    RPN_NMS_THRESHOLD = 0.9
    MEAN_PIXEL = np.array([105, 236, 189])
    DETECTION_NMS_THRESHOLD = 0.3
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
# end Class OrangeCanopyConfig

class OrangeCanopyDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, dataset_dir, subset):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """

        assert subset in ['train', 'val', 'real_test']
        dataset_dir = os.path.join(dataset_dir, subset)

        annotation_json = os.path.join(dataset_dir, 'coco_annotations.json')
        images_dir = os.path.join(dataset_dir, 'images')

        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be le500ss than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    # End load_data

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids
    # end load_mask

# end class OrangeCanopyDataset

def train(model):
    # Training dataset.
    train_dataset = OrangeCanopyDataset()
    train_dataset.load_data(args.dataset, 'train')
    train_dataset.prepare()

    # Validation dataset
    global val_dataset
    val_dataset = OrangeCanopyDataset()
    val_dataset.load_data(args.dataset, 'val')
    val_dataset.prepare()

    augmentation = iaa.SomeOf((0, 2), [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.OneOf(
        [iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]), iaa.Multiply((0.5, 1.5))])

    # You can first train the heads
    print('Training Heads...')
    model.train(train_dataset, val_dataset,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                augmentation=augmentation,
                layers='heads')

    # and after, You can train all the neural network
    # see parameters.txt to get hints

    # print('Training All One...')
    # model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE, epochs=40, augmentation=augmentation, layers='all')

    # print('Training All Two...')
    # model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE/10, epochs=40, augmentation=augmentation, layers='all')

    # print('Training All Tree...')
    # model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE/100, epochs=40, augmentation=augmentation, layers='all')

    # print('Training All Four...')
    # model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE/1000, epochs=40, augmentation=augmentation, layers='all')

    return 'Training Finished!!!!'

def prediction(model, imgs_dir):      # Trabalhar nisso aqui!!! #######################################################

    shape_path_aux = args.dataset + '/result/'
    temp_path_prediction = args.dataset + '/result/' + 'TEMP_RESP.tif'
    shape_prediction_name = 'pred_image'
    shape_prediction_ext = '.shp'
    dataset_prediction = OrangeCanopyDataset()
    dataset_prediction.load_trees(args.dataset, 'prediction', args.pred_images)
    dataset_prediction.prepare()

    # código do ypnb canopyTrainAndInference
    import skimage
    real_test_dir = imgs_dir
    image_paths = []
    for filename in os.listdir(real_test_dir):
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(os.path.join(real_test_dir, filename))

    for image_path in image_paths:
        array, image_tif = gdb.readimagetif(image_path, 'Integer')
        results = model.detect([array], verbose=1)[0]
        masks = result['masks']
        dim_masks = masks.shape
        masks_final = np.zeros((dim_masks[0], dim_masks[1]), dtype=np.int32)
        for cont_mask in range(dim_masks[2]):
            aux = masks[:, :, cont_mask].copy()
            for j in range(aux.shape[0]):
                for i in range(aux.shape[1]):
                    if aux[j, i] != 0:
                        masks_final[j, i] = (cont_mask + 1)

        # save the shape
        origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height = image_tif.GetGeoTransform()
        drive = image_tif.GetDriver()
        projection = image_tif.GetProjection()
        raster_origin = (origin_y, origin_x)
        gdb.array2raster(temp_path_prediction, 'Integer', raster_origin, pixel_height, pixel_width, rot_y, rot_x, drive,
                         projection, masks_final)
        shape_path = shape_path_aux + shape_prediction_name + str(k) + shape_prediction_ext
        gdb.raster2polygon(temp_path_prediction, shape_path, 'result')
        masks_final = None

        r = results[0]
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                    val_dataset.class_names, r['scores'], figsize=(5, 5))


    print('Prediction Done')


if __name__ == '__main__':
    import argparse

    ## Parse command line arguments
    parser = argparse.ArgumentParser(description="Train -- Mask R-CNN to detect orange's trees canopy.")
    parser.add_argument('command', metavar='<command>', help="'train', or 'prediction'")
    parser.add_argument('--dataset', required=True, metavar="/path/to/tree/dataset/",
                        help='Directory of the Trees dataset')
    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False, default=MODEL_DIR, metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()

    ## Validate arguments
    if args.command == 'train':
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == 'prediction':
        assert args.dataset, "Argument --dataset is required for prediction"
    else:
        assert args.dataset, "Argument --dataset is required"

    print('Weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print('Logs: the weights are stored at', args.logs)


    ## Configurations
    if args.command == "train":
        config = OrangeCanopyConfig()
    else:
        class InferenceConfig(OrangeCanopyConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            # I kept the same configuration of the training
            NUM_CLASSES = 1 + 1
            NAME = 'trees'
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            BACKBONE = "resnet50"
            DETECTION_MIN_CONFIDENCE = 0.5
            BACKBONE_STRIDES = [4, 8, 16, 32, 64]
            RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
            USE_MINI_MASK = False
            IMAGE_CHANNEL_COUNT = 3
            RPN_NMS_THRESHOLD = 0.9
            MEAN_PIXEL = np.array([105, 236, 189])
            DETECTION_NMS_THRESHOLD = 0.3
            IMAGE_RESIZE_MODE = "square"
            IMAGE_MIN_DIM = 512
            IMAGE_MAX_DIM = 512
            MAX_GT_INSTANCES = 100
            DETECTION_MAX_INSTANCES = 100
            TRAIN_ROIS_PER_IMAGE = 100
            RPN_TRAIN_ANCHORS_PER_IMAGE = 100


        # end InferenceConfig()
        config = InferenceConfig()
    # end else
    config.display()

    # Create model
    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=args.logs)

    print("Loading weights ", args.weights)

    # Select weights file to load
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(args.weights, by_name=True)

    # Train or evaluate
    if args.command == 'train':
        train(model)
    elif args.command == 'prediction':
        images_test_dir = os.path.join(args.dataset, 'real_test', 'images')
        prediction(model, images_test_dir)
    else:
        print("'{}' this is not recognized. Use 'train' or prediction'".format(args.command))
