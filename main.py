#!/usr/bin/env python

import argparse
import logging
from src.parse_data import ParseData


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
    parser.add_argument('--background_proportion', default=0.2, type=float, metavar='BACKGROUND_PROPORTION',
                        help="The proportion of the balanced dataset that should be background patches without mitochondria. "
                             "If more background patches than what is available in the dataset are requested, the maximum available "
                             "in the dataset will be used unless the '--background_augment' flag is used.")
    parser.add_argument('--background_augment', action='store_false',
                        help="Set this if you want to perform data augmentation on background patches to get to the proportion "
                             "set in the '--background_proportion' parameter if there are not enough background patches in the "
                             "dataset without augmentation. If there are enough, this will be ignored and no augmentation will "
                             "be performed. These are simple rotations that will not leave a feature for the model to learn."
                             "This is enabled by default.")

    args = parser.parse_args()

    return args


def main():
    # configure logging
    logging.basicConfig(filename='mitochondrial_segmentation.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting main script...')

    args = parse()

    data_parser = ParseData(args.patch_height, args.patch_width, args.stride, args.pairs, args.background_proportion,
                            args.background_augment)
    data_parser.collect_data(args.download)


if __name__ == '__main__':
    main()
