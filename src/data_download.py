import logging
import os
import numpy as np
import requests
from tqdm import tqdm
import tifffile


class DownloadData:
    def __init__(self, path='mitochondrial_EM_image.tif'):
        self.filename = os.path.join('images', path)
        self.images = np.empty((1065, 1536, 2048), dtype=np.uint8)
        self.data_source = 'https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/volumedata.tif'
        self.logger = logging.getLogger(__name__)

    def __download(self):
        # set variables
        block_size = 1024

        print('downloading…')
        # send request
        response = requests.get(self.data_source, stream=True)
        response.raise_for_status()
        size = int(response.headers.get('content-length', 0))

        with open(self.filename, 'wb') as file, tqdm(
                desc=self.filename,
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

    def __get_images(self):
        image_shape = (1536, 2048)

        with tifffile.TiffFile(self.filename) as tif:
            for i, page in enumerate(tif.pages):
                try:
                    image = page.asarray()
                    self.logger.info(
                        f"Page {i + 1} loaded with shape: {image.shape}, dtype: {image.dtype}, size: {image.nbytes / 1e6:.2f} MB")
                    assert image.shape == image_shape, f'Unexpected shape at page {i}'
                    self.images[i] = image

                except Exception as e:
                    self.logger.error(f'Page {i + 1} failed: {e}')

    def collect_data(self):
        self.__download()
        self.__get_images()
