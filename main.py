#!/usr/bin/env python

import argparse
import logging
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from src.parse_data import ParseData
from src.model_train import MitochondrialEMDataset, UNet, train_model, test_model


def parse():
    parser = argparse.ArgumentParser(
        description='Train a model to perform semantic segmentation from EM images and then can run the model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--download', action='store_true',
                        help='Set this is you need to download the data. Please only perform this once as needed to not '
                             'needless use the host site.')
    parser.add_argument('--patch_height', default=256, type=int,
                        help='The height of patches from the original (768, 1024) image.')
    parser.add_argument('--patch_width', default=256, type=int,
                        help='The width of patches from the original (768, 1024) image.')
    parser.add_argument('--stride', default=256, type=int,
                        help='The stride when patching the large image determining how much overlap there is between patches. '
                             'Default is no overlap.')
    parser.add_argument('--pairs', default=5, type=int,
                        help='The number of image/mask pairs to plot during basic data exploration.')
    parser.add_argument('--background_proportion', default=0.2, type=float,
                        help="The proportion of the balanced dataset that should be background patches without mitochondria. "
                             "If more background patches than what is available in the dataset are requested, the maximum available "
                             "in the dataset will be used unless the '--background_augment' flag is used.")
    parser.add_argument('--background_augment', action='store_false',
                        help="Set this if you want to perform data augmentation on background patches to get to the proportion "
                             "set in the '--background_proportion' parameter if there are not enough background patches in the "
                             "dataset without augmentation. If there are enough, this will be ignored and no augmentation will "
                             "be performed. These are simple rotations that will not leave a feature for the model to learn."
                             "This is enabled by default.")
    parser.add_argument('--train_proportion', default=0.8, type=float,
                        help='The proportion of the total data that should be used for training. The testing proportion'
                             'will be 1 minus this value.')
    parser.add_argument('--validation_proportion', default=0.1, type=float,
                        help='The proportion of the training split that will be reserved for validation.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='The batch size to be used for training.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='The number of training epochs.')
    parser.add_argument('--test_examples', default=3, type=int,
                        help='The number of examples to depict from the test dataset.')

    args = parser.parse_args()

    return args


def main():
    # configure logging
    logging.basicConfig(filename='mitochondrial_segmentation.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting main script...')

    args = parse()
    logging.info("Arguments: %s", vars(args))

    # handle the data and prepare it for input into the model
    data_parser = ParseData(args.patch_height, args.patch_width, args.stride, args.pairs, args.background_proportion,
                            args.background_augment, args.train_proportion, args.validation_proportion)
    data_parser.collect_data(args.download)

    if args.validation_proportion > 0:
        # create the datasets
        train_dataset = MitochondrialEMDataset(data_parser.train_split['images'], data_parser.train_split['masks'])
        val_dataset = MitochondrialEMDataset(data_parser.val_split['images'], data_parser.val_split['masks'])
        test_dataset = MitochondrialEMDataset(data_parser.test_split['images'], data_parser.test_split['masks'])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        # collect examples to plot
        example_idx = random.sample(range(len(test_dataset)), args.test_examples)
        example_images = [test_dataset[i] for i in example_idx]


        model = UNet()
        history = train_model(model, train_loader, args.epochs, logging, val_loader)
        test_model(model, test_loader, logging, example_images)

        # Loss Plot
        plt.figure(figsize=(8, 5))
        plt.plot(list(range(1, args.epochs + 1)), history['train_loss'], label='Train Loss')
        plt.plot(list(range(1, args.epochs + 1)), history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('images/loss_plot.png')
        plt.close()

        # Accuracy Plot
        plt.figure(figsize=(8, 5))
        plt.plot(list(range(1, args.epochs + 1)), history['train_acc'], label='Train Accuracy')
        plt.plot(list(range(1, args.epochs + 1)), history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig('images/accuracy_plot.png')
        plt.close()

        # IoU Plot
        plt.figure(figsize=(8, 5))
        plt.plot(list(range(1, args.epochs + 1)), history['train_acc'], label='Train IoU')
        plt.plot(list(range(1, args.epochs + 1)), history['val_acc'], label='Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('Training vs Validation IoU')
        plt.legend()
        plt.grid(True)
        plt.savefig('images/iou_plot.png')
        plt.close()
    else:
        # create the datasets
        train_dataset = MitochondrialEMDataset(data_parser.train_split['images'], data_parser.train_split['masks'])
        test_dataset = MitochondrialEMDataset(data_parser.test_split['images'], data_parser.test_split['masks'])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        # collect examples to plot
        example_idx = random.sample(range(len(test_dataset)), args.test_examples)
        example_images = [test_dataset[i] for i in example_idx]

        model = UNet()
        history = train_model(model, train_loader, args.epochs, logging)
        test_model(model, test_loader, logging, example_images)

        # Loss Plot
        plt.figure(figsize=(8, 5))
        plt.plot(list(range(1, args.epochs + 1)), history['train_loss'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('images/loss_plot.png')
        plt.close()

        # Accuracy Plot
        plt.figure(figsize=(8, 5))
        plt.plot(list(range(1, args.epochs + 1)), history['train_acc'], label='Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig('images/accuracy_plot.png')
        plt.close()

        # IoU Plot
        plt.figure(figsize=(8, 5))
        plt.plot(list(range(1, args.epochs + 1)), history['train_acc'], label='Train IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('Training IoU')
        plt.legend()
        plt.grid(True)
        plt.savefig('images/iou_plot.png')
        plt.close()


if __name__ == '__main__':
    main()
