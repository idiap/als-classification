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
from src import utils


def test_loaders():
    protocol = Protocol(['control', 'als'], ['oxidative'], 'hnRNPK')
    classification = 'als'
    channels = ['DAPI']
    fold = 0
    config = utils.get_config('user')
    database = Database()

    train, test = database.cross_validation(classification, protocol, fold)
    train_loader = utils.create_train_dataloader(config, database, train, classification, protocol, channels, fold)
    test_loader, test_dataset = utils.create_test_dataloader(config, database, test, classification, protocol, channels,
                                                             fold)
    assert len(train_loader) == len(train) * 16 * 6 / config['batch_size']
    assert len(test_loader) == len(test) * 16 / config['batch_size']

    test_dataset.delete()
