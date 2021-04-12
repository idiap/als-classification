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

from src.database import Database
from src.protocol import Protocol
from src.dataset import CustomDataset
from src import utils

import torchvision.transforms as transforms
import pytest
import os

database = Database()
protocol = Protocol(['control', 'als'], ['untreated'], 'TDP-43')
channels = ['DAPI', 'BIII', 'TDP-43']
fold = 0
classification = 'als'
train_indices, test_indices = database.cross_validation('als', protocol, fold)
basic_data_transform = transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
config = utils.get_config('user')
train_dataset = CustomDataset(config, database, train_indices, basic_data_transform, classification, protocol, channels,
                              fold)
test_dataset = CustomDataset(config, database, test_indices, basic_data_transform, classification, protocol, channels,
                             fold)


@pytest.mark.parametrize("input_,output_",
                         [
                             ((2, 4, 1, 1, 4), '002004-1-001001004'),
                             ((2, 10, 1, 1, 4), '002010-1-001001004'),
                         ]
                         )
def test_get_image_id(input_, output_):
    assert CustomDataset.get_image_id(*input_) == output_


@pytest.mark.parametrize("input_,output_", [(('screenE', 'P1', '001-002'), 'screenE/P1/001-002.tif'), ])
def test_get_filename(input_, output_):
    actual = train_dataset.get_filename(*input_)
    assert actual == f'{train_dataset.source_directory}/{output_}'


def test_init():
    assert not train_dataset.df.empty
    assert not test_dataset.df.empty
    assert len(train_dataset) == 9904
    assert len(test_dataset) == 1104


@pytest.mark.parametrize("sample", [(train_dataset.df.loc[0])])
def test_image_methods(sample):
    path = os.path.join(utils.get_project_root(), 'data/test_images')
    if not os.path.exists(path):
        os.mkdir(path)
    images = dict()
    for channel in channels:
        mip_image = train_dataset.maximum_intensity_projection(sample, channel)
        mip_image.save(f'{path}/mip_image_{channel}.tif')
        images[channel] = mip_image

    fused_image = train_dataset.fuse_channels(images)
    fused_image.save(f'{path}/fused_image.tif')
    processed_image = train_dataset.process_sample(sample, channels)
    processed_image.save(f'{path}/processed_image.tif')
    assert processed_image == fused_image

    cropped_images = train_dataset.crop_image(processed_image, 270)
    for i, crop_image in enumerate(cropped_images):
        crop_image.save(f'{path}/crop_{i}.tif')


train_dataset.delete()  # also deletes test_dataset folder because train and test share a folder
