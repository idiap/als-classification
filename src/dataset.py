# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Colombine Verzat <colombine.verzat@idiap.ch>

# This file is part of als-classification.
#
# als-classification is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# als-classification is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with als-classification. If not, see <http://www.gnu.org/licenses/>.

import os
import pandas as pd
import skimage
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import warnings
from src import utils
import shutil
from typing import List, Dict


class CustomDataset(Dataset):
	"""Define a custom dataset which can be used with Pytorch"""

	def __init__(self, config, database, indices, transform, classification, protocol, channels, fold):
		"""Initialize a CustomDataset

		Args:
			config: yaml file with your configuration
			indices (list[int]): which indices of the protocol you want to use
			transform (torchvisions.transforms): transform which will be applied to all images
			classification (str): either `als` or `stress`
			protocol (Protocol): which protocol is used
			channels (list[str]): selected fluorescent markers
			fold (int): which fold of cross-validation
		"""
		self.source_directory = config['source_directory']
		self.target_directory = os.path.join(config['target_directory'], f'{protocol.name}_{"_".join(channels)}_{fold}')
		if not os.path.exists(self.target_directory):
			os.mkdir(self.target_directory)
		try:
			path = os.path.join(utils.get_project_root(), 'data/protocols', f'{protocol.name}.csv')
			protocol_data = pd.read_csv(path)
		except FileNotFoundError:
			protocol_data = database.get_protocol_data(protocol, save=True)

		self.df = protocol_data.loc[indices]
		self.df = self.df.reset_index(drop=True)  # Pytorch dataset needs to access elements starting from 0

		if classification == 'stress':
			self.df['y'] = self.df['stress_label'].map({'stress': 1, 'no_stress': 0})
		elif classification == 'als':
			self.df['y'] = self.df['als_label'].map({'als': 1, 'control': 0})
		self.df['hash'] = self.df.apply(lambda x: hash(tuple(x)), axis=1)

		self.channels = channels
		self.transform = transform

	def __len__(self):
		"""Number of images in the dataset"""
		return len(self.df) * 16  # each image is divided in 16 smaller images

	def __getitem__(self, idx: int) -> (Image.Image, int, int):
		"""Get one image from the dataset"""
		idx_original_image = int(idx / 16)
		idx_crop = idx % 16
		sample = self.df.loc[idx_original_image]
		filename = f'{self.target_directory}/{sample.hash}-{idx_crop}.tif'
		try:
			image = Image.open(filename)
		except FileNotFoundError:
			image = self.process_sample(sample, self.channels)
			if self.check_empty_image(image):
				return None, sample.y, idx_original_image
			cropped_images = self.crop_image(image, chopsize=270)
			for crop in range(16):
				filename = f'{self.target_directory}/{sample.hash}-{crop}.tif'
				cropped_images[crop].save(filename)
			image = cropped_images[idx_crop]
		image = self.transform(image)
		return image, sample.y, idx_original_image

	def access_all_elements(self) -> (Image.Image, int, int):
		"""Access each element in the dataset"""
		for i in range(self.__len__()):
			self.__getitem__(i)

	def get_filename(self, experiment: str, plate: str, image_id: str) -> str:
		"""Return pathname of an image"""
		return f'{self.source_directory}/{experiment}/{plate}/{image_id}.tif'

	@staticmethod
	def get_image_id(row: int, col: int, fov: int, plane: int, channel: int) -> str:
		"""Return id of an image"""
		return f'{row:03}{col:03}-{fov}-001{plane:03}{channel:03}'

	def process_sample(self, sample: pd.Series, channels: List[str]) -> Image.Image:
		"""Do maximum intensity projection and fuse channels of the resulting images"""
		images = dict()
		for channel in channels:
			images[channel] = self.maximum_intensity_projection(sample, channel)
		fused_image = self.fuse_channels(images)
		return fused_image

	def maximum_intensity_projection(self, sample: pd.Series, channel: List[str]) -> Image.Image:
		"""Take maximum pixel intensity across z-stacks (planes)"""
		channel_nb = list(sample.filter(regex=r'channel_\d').values).index(
			channel) + 1  # find number corresponding to the channel
		images = np.zeros((sample.number_of_planes, 1080, 1080), dtype=np.uint16)
		for p in range(sample.number_of_planes):
			image_id = self.get_image_id(sample.well_row, sample.well_col, sample.fov, p + 1, channel_nb)
			filename = self.get_filename(sample.experiment, sample.plate, image_id)
			images[p, :, :] = np.asarray(Image.open(filename))
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			mip_image = Image.fromarray(skimage.util.img_as_ubyte(np.max(images, axis=0)))
		return mip_image

	@staticmethod
	def fuse_channels(images: Dict[str, Image.Image]) -> Image.Image:
		"""Assign a channel to each RGB field"""
		channels = list(images.keys())
		images['empty'] = Image.fromarray(np.zeros((1080, 1080), dtype='uint8'))
		if len(channels) == 2:  # if there are only 2 channels, add empty channel
			fused_image = Image.merge("RGB", [images[channels[1]], images['empty'], images[channels[0]]])
		elif len(channels) == 1:  # if there is only 1 channel, add 2 empty channels
			fused_image = Image.merge("RGB", [images[channels[0]], images['empty'], images['empty']])
		else:  # assign each channel to one RGB field
			fused_image = Image.merge("RGB", [images[channels[1]], images[channels[2]], images[channels[0]]])
		fused_image = ImageOps.autocontrast(fused_image, cutoff=0.1)  # enhance image contrast
		return fused_image

	@staticmethod
	def crop_image(image: Image.Image, chopsize: int) -> List[Image.Image]:
		"""Split image in smaller images of chopsize*chopsize"""
		cropped_images = []
		width, height = image.size
		for x0 in range(0, width, chopsize):
			for y0 in range(0, height, chopsize):
				box = (x0, y0, x0 + chopsize if x0 + chopsize < width else width - 1, y0 + chopsize if y0 + chopsize < height else height - 1)
				cropped_images.append(image.crop(box))
		return cropped_images

	def check_empty_image(self, image: Image.Image) -> bool:
		"""Get average pixel intensity and percentage of black pixels in the image"""

		min_red = list(image.getdata(0)).count(0) / len(
			image.getdata(0))  # count number of zeros in first (red) band and compute percentage
		min_green = list(image.getdata(1)).count(0) / len(image.getdata(0))  # same for green band
		min_blue = list(image.getdata(2)).count(0) / len(image.getdata(0))  # same for blue band

		if len(self.channels) == 1:  # only red channel is used
			return min_red > 0.99
		elif len(self.channels) == 2:  # red and blue channels are used
			return min_red > 0.99 or min_blue > 0.99
		elif len(self.channels) == 3:  # all channels are used
			return min_red > 0.99 or min_green > 0.99 or min_blue > 0.99

	def delete(self):
		"""Delete the folder where the dataset was created"""
		shutil.rmtree(self.target_directory, ignore_errors=True)  # delete directory
