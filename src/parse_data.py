import logging
import os
import requests
from tqdm import tqdm
import tifffile
import matplotlib.pyplot as plt
import random
import numpy as np


class ParseData:
    def __init__(self, height, width, stride, pairs, background_proportion, background_augment, train_prop, validation_prop):
        self.images = {'images': [],
                       'masks': []
                       }
        self.patches = {'images': [],
                        'masks': []
                        }
        self.balanced_patches = {'images': [],
                                 'masks': []
                                 }
        self.train_split = {'images': [],
                            'masks': []
                            }
        self.val_split = {'images': [],
                          'masks': []}
        self.test_split = {'images': [],
                           'masks': []
                           }
        self.height = height
        self.width = width
        self.stride = stride
        self.pairs = pairs
        self.background_proportion = background_proportion
        self.background_augment = background_augment
        self.train_prop = train_prop
        self.val_prop = validation_prop
        self.logger = logging.getLogger(__name__)

    def __download(self, source, filename):
        """
        Downloads the data using the provided source url, and saves it in the images directory.
        :param source: The url to the data download, string
        :param filename: The name to save the data download as in the images directory, string
        :return: None
        """
        filename = os.path.join('images', filename)
        self.logger.info(
            'Starting the download of the data from {source} into {filename}.')

        # set variables
        block_size = 1024

        # send request
        response = requests.get(source, stream=True)
        response.raise_for_status()
        size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as file, tqdm(
                desc=filename,
                total=size,
                unit='iB',
                unit_scale=True,
                unit_divisor=block_size
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                file.write(chunk)
                bar.update(len(chunk))

            # make sure everything is properly flushed to disk
            file.flush()
            os.fsync(file.fileno())

    def __get_images(self, filename, image_set):
        """
        Retrieves the individual images from a multipage .tif from the filename provided. Stores them in the dictionary
        self.images under the provided image_set key
        :param filename: name of the .tif file, string
        :param image_set: key for the self.images dictionary, string
        :return: None
        """
        filename = os.path.join('images', filename)
        self.logger.info(f'reading in the images from {filename} to the {image_set} dataset')
        image_shape = (768, 1024)

        if not os.path.exists(filename):
            self.logger.error(
                f"The file '{filename}' is not present. Make sure you have run this with the '--download' flag to download the data.")
            raise FileExistsError(
                f"The file '{filename}' is not present. Make sure you have run this with the '--download' flag to download the data.")

        with tifffile.TiffFile(filename) as tif:
            for i, page in enumerate(tif.pages):
                try:
                    image = page.asarray()
                    self.logger.info(
                        f"Page {i + 1} loaded with shape: {image.shape}, dtype: {image.dtype}, size: {image.nbytes / 1e6:.2f} MB")
                    assert image.shape == image_shape, f'Unexpected shape at page {i}'
                    self.images[image_set].append(image)

                except Exception as e:
                    self.logger.error(f'Page {i + 1} failed: {e}')

    def __patch_images(self):
        """
        Split the images into patches
        :return: None
        """
        for x, y in zip(self.images['images'], self.images['masks']):
            for i in range(0, x.shape[0] - self.height + 1, self.stride):
                for j in range(0, x.shape[1] - self.width + 1, self.stride):
                    self.patches['images'].append(x[i:i + self.height, j:j + self.width])
                    self.patches['masks'].append(y[i:i + self.height, j:j + self.width])

    def __balance_data(self):
        """
        Balance the amount of background images to non-background images to align with the input parameters.
        :return: None
        """
        background_images_x = []
        background_images_y = []
        for x, y in zip(self.patches['images'], self.patches['masks']):
            if np.sum(y) != 0:
                self.balanced_patches['images'].append(x)
                self.balanced_patches['masks'].append(y)
            else:
                background_images_x.append(x)
                background_images_y.append(y)

        num_background = int(len(self.balanced_patches['images']) / (1 - self.background_proportion)) - len(
            self.balanced_patches['images'])

        # add the amount of background images. Augment the dataset if requested
        if num_background >= len(background_images_x) and not self.background_augment:
            for idx in range(len(background_images_x)):
                self.balanced_patches['images'].append(background_images_x[idx])
                self.balanced_patches['masks'].append(background_images_y[idx])

        elif num_background > len(background_images_x) and self.background_augment:
            num_patches_needed = num_background - len(background_images_x)
            print(f'\nBackground augment was set, and {num_patches_needed} patches will be created.')
            self.logger.info(f'Background augment was set, and {num_patches_needed} patches will be created.')

            add_x, add_y = self.__rotate(background_images_x, background_images_y, num_patches_needed)
            background_images_x.extend(add_x)
            background_images_y.extend(add_y)
            for idx in range(len(background_images_x)):
                self.balanced_patches['images'].append(background_images_x[idx])
                self.balanced_patches['masks'].append(background_images_y[idx])

        elif num_background <= len(background_images_x) and self.background_augment:
            print(
                f'\nBackground augment was set, but no additional patches are needed. No augmentation will be performed.')
            self.logger.info(
                f'Background augment was set, but no additional patches are needed. No augmentation will be performed.')
            background_idx = random.sample(range(len(background_images_x)), num_background)
            for idx in background_idx:
                self.balanced_patches['images'].append(background_images_x[idx])
                self.balanced_patches['masks'].append(background_images_y[idx])

        else:
            background_idx = random.sample(range(len(background_images_x)), num_background)
            for idx in background_idx:
                self.balanced_patches['images'].append(background_images_x[idx])
                self.balanced_patches['masks'].append(background_images_y[idx])

    def __explore_data(self):
        """
        Report some features of the data and create a small demo figure of some image and mask pairs from the data.
        :return: None
        """
        # calculate amount of background images
        background = 0
        mito_images = 0
        for image in self.patches['masks']:
            if np.sum(image) == 0:
                background += 1
            else:
                mito_images += 1
        print(f'\nThere are {background + mito_images} total patches in the dataset')
        print(f'There are {mito_images} patches with mitochondria before balancing.')
        print(f'There are {background} background patches in the dataset before balancing.')
        self.logger.info(f'There are {background + mito_images} total patches in the dataset')
        self.logger.info(f'There are {mito_images} patches with mitochondria before balancing.')
        self.logger.info(f'There are {background} background patches in the dataset before balancing.')

        # calculate from balanced dataset
        balanced_background = 0
        balanced_mito = 0
        for image in self.balanced_patches['masks']:
            if np.sum(image) == 0:
                balanced_background += 1
            else:
                balanced_mito += 1

        print(f'\nThere are a total of {balanced_background + balanced_mito} patches in the balanced dataset.')
        print(f'There are {balanced_mito} patches with mitochondria in the balanced dataset.')
        print(f'There are {balanced_background} background patches in the balanced dataset.')
        print(
            f'The proportion of background images in the balanced dataset is {balanced_background / (balanced_background + balanced_mito):.2f}')
        self.logger.info(f'There are a total of {balanced_background + balanced_mito} patches in the balanced dataset.')
        self.logger.info(f'There are {balanced_mito} patches with mitochondria in the balanced dataset.')
        self.logger.info(f'There are {balanced_background} background patches in the balanced dataset.')
        self.logger.info(
            f'The proportion of background images in the balanced dataset is {balanced_background / (balanced_background + balanced_mito):.2f}')

        if self.val_prop > 0:
            print(f"\nThere are {len(self.train_split['images'])} patches in the train set.")
            print(f"There are {len(self.val_split['images'])} patches in the validation set.")
            print(f"There are {len(self.test_split['images'])} patches in the test set.")
            self.logger.info(f"There are {len(self.train_split['images'])} patches in the train set.")
            self.logger.info(f"There are {len(self.val_split['images'])} patches in the validation set.")
            self.logger.info(f"There are {len(self.test_split['images'])} patches in the test set.")
        else:
            print(f"\nThere are {len(self.train_split['images'])} patches in the train set.")
            print(f"There are {len(self.test_split['images'])} patches in the test set.")
            self.logger.info(f"There are {len(self.train_split['images'])} patches in the train set.")
            self.logger.info(f"There are {len(self.test_split['images'])} patches in the test set.")

        # plot the specified pairs
        plot_images = random.sample(range(int(len(self.balanced_patches['images']) * self.background_proportion)),
                                    self.pairs)
        fig, axes = plt.subplots(self.pairs, 2, figsize=(6, 1.5 * self.pairs))

        if self.pairs == 1:
            axes = [axes]

        for row, i in enumerate(plot_images):
            image = self.balanced_patches['images'][i]
            mask = self.balanced_patches['masks'][i]

            ax1, ax2 = axes[row]
            ax1.imshow(image)
            ax1.set_title('Image')
            ax1.axis('off')

            ax2.imshow(mask, cmap='viridis')
            ax2.set_title('Mask')
            ax2.axis('off')

        plt.tight_layout()
        plt.savefig('images/image_and_mask_demo.png')

    @staticmethod
    def __rotate(images, masks, needed):
        """
        Rotate the number of specified images and masks.
        :param images: List of all possible images to rotate, list
        :param masks: List of all possible masks to rotate, list
        :param needed: Number of image/mask pairs to rotate, int
        :return: rotated images and masks, tuple (images, masks)
        """
        angles = [1, 2, 3]
        used = set()  # set to track to make sure there are no duplicates
        additional_images = []
        additional_masks = []

        for i in range(needed):
            idx = random.sample(range(len(images)), 1)[0]
            angle = random.sample(angles, 1)[0]

            while f'{angle},{idx}' in used:  # make sure there are no duplicate patches created. Can get stuck here if a lot of images are asked for
                idx = random.sample(range(len(images)), 1)[0]
                angle = random.sample(angles, 1)[0]

            used.add(f'{angle},{idx}')
            additional_images.append(np.rot90(images[idx], angle))
            additional_masks.append(np.rot90(masks[idx], angle))

        return additional_images, additional_masks

    def __split_data(self):
        data_size = len(self.balanced_patches['images'])
        num_train = int(self.train_prop * data_size)
        train_idxs = set(random.sample(range(data_size), num_train))
        if self.val_prop > 0:
            num_val = int(self.val_prop * num_train)
            val_idxs = set(random.sample(list(train_idxs), num_val))

            for i in range(data_size):
                if i in val_idxs:
                    self.val_split['images'].append(self.balanced_patches['images'][i])
                    self.val_split['masks'].append(self.balanced_patches['masks'][i])
                elif i in train_idxs:
                    self.train_split['images'].append(self.balanced_patches['images'][i])
                    self.train_split['masks'].append(self.balanced_patches['masks'][i])
                else:
                    self.test_split['images'].append(self.balanced_patches['images'][i])
                    self.test_split['masks'].append(self.balanced_patches['masks'][i])
        else:
            for i in range(data_size):
                if i in train_idxs:
                    self.train_split['images'].append(self.balanced_patches['images'][i])
                    self.train_split['masks'].append(self.balanced_patches['masks'][i])
                else:
                    self.test_split['images'].append(self.balanced_patches['images'][i])
                    self.test_split['masks'].append(self.balanced_patches['masks'][i])

    def collect_data(self, download):
        """
        Driver of the class
        :param download: state whether the data needs to be downloaded, bool
        :return: None
        """
        data_sources = [
            'https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training.tif',
            'https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training_groundtruth.tif',
            'https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/testing.tif',
            'https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/testing_groundtruth.tif'
        ]

        if download:
            self.__download(data_sources[0], data_sources[0].split('/')[-1].replace('training', 'set1'))
            self.__download(data_sources[1], data_sources[1].split('/')[-1].replace('training', 'set1'))
            self.__download(data_sources[2], data_sources[2].split('/')[-1].replace('testing', 'set2'))
            self.__download(data_sources[3], data_sources[3].split('/')[-1].replace('testing', 'set2'))

        self.__get_images(data_sources[0].split('/')[-1].replace('training', 'set1'), 'images')
        self.__get_images(data_sources[1].split('/')[-1].replace('training', 'set1'), 'masks')
        self.__get_images(data_sources[2].split('/')[-1].replace('testing', 'set2'), 'images')
        self.__get_images(data_sources[3].split('/')[-1].replace('testing', 'set2'), 'masks')

        self.__patch_images()
        self.__balance_data()

        self.__split_data()

        self.__explore_data()
