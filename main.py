#!/usr/bin/env python

import argparse
import logging
from src.data_download import DownloadData


def parse():
    parser = argparse.ArgumentParser(
        description='Train a model to perform semantic segmentation from EM images and then can run the model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--download', action='store_true', default=argparse.SUPPRESS,
                        help='Set this is you need to download the data.')
    parser.add_argument('--raw_download_file', default='mitochondrial_EM_image.tif',
                        help='Set the name of the multipage tif download. It will be saved in the ./scr directory.')

    args = parser.parse_args()

    return args


def main():
    # configure logging
    logging.basicConfig(filename='mitochondrial_segmentation.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting main script...')

    args = parse()

    if args.download:
        downloader = DownloadData()
        downloader.collect_data()


if __name__ == '__main__':
    main()
