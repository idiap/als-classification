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

import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import click
import os


@click.command()
@click.option('-c', '--config_name', type=str, required=True, help='''Name of the configuration file''')
@click.option('-cl', '--classification', type=click.Choice(['stress', 'als']), required=True,
              help='''Classify stress vs untreated (choose 'stress') or als vs control (choose 'als')''')
@click.option('-p', '--protocol_name', type=str, required=True,
              help='''Name of the protocol for training (available protocols are listed in database.protocols)''')
@click.option('-ch', '--channels', type=click.Choice(['DAPI', 'BIII', 'SFPQ', 'FUS', 'TDP-43', 'hnRNPA1', 'hnRNPK']),
              required=True, multiple=True, help='''List of channels used for training''')
@click.option('-f', '--fold', type=int, required=True, help='''Fold for 10-fold cross validation (between 0 and 9)''')
@click.option('-s', '--save_state_dict', type=bool, help='''Save state dict of the trained model''')
@click.option('--dry_run', is_flag=True, help="Dry run for testing")
def classify(config_name, classification, protocol_name, channels, fold, save_state_dict=False, dry_run=False):
    """Train a model"""

    # get configuration parameters
    config = utils.get_config(config_name)
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # initialize mobilenet pre-trained model
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # binary classification

    # get protocol data and indices for cross-validation
    database = Database()
    protocol = Protocol.from_name(protocol_name)
    title = f'{protocol_name}_{"_".join(channels)}_fold_{fold}'
    train, test = database.cross_validation(classification, protocol, fold)

    # create dataloaders and initialize criterion, optimizer
    train_loader = utils.create_train_dataloader(config, database, train, classification, protocol, channels, fold)
    test_loader, test_dataset = utils.create_test_dataloader(config, database, test, classification, protocol, channels,
                                                             fold)
    criterion, optimizer = utils.init_model(model, device, learning_rate)

    if not dry_run:
        # train and evaluate performance with AUC
        test_auc = utils.train_and_evaluate(n_epochs, model, train_loader, test_loader, criterion, optimizer, device,
                                            title, save_state_dict=save_state_dict, debug=False)

        if not save_state_dict:
            # write results in auc.csv
            path = os.path.join(utils.get_project_root(), f'results/auc.csv')
            results = pd.read_csv(path, index_col='protocol')
            results[f'fold_{fold}'].loc[f'{protocol_name}_{"_".join(channels)}'] = test_auc
            results.to_csv(path, index_label='protocol')

    # delete folder with images
    test_dataset.delete()


if __name__ == "__main__":
    classify()
