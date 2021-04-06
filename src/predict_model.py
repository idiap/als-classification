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
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import click
import warnings
import os
from typing import List

CLASSIFIER_NAMES = {
    'als': 'control_als_untreated',
    'osmotic': 'control_untreated_osmotic',
    'heat': 'control_untreated_heat',
    'oxidative': 'control_untreated_oxidative'
}


@click.command()
@click.option('-c', '--config_name', type=str, required=True,
              help='''Name of the configuration file''')
@click.option('-cl', '--classifier', type=click.Choice(['als', 'osmotic', 'oxidative', 'heat']), required=True,
              help='''Choose classifier: als vs control (choose 'als'), osmotic vs untreated (choose 'osmotic'), oxidative 
                vs untreated (choose 'oxidative') or heat vs untreated (choose 'heat')''')
@click.option('-e', '--expert',
              type=click.Choice(['DAPI', 'BIII', 'DAPI_BIII', 'SFPQ', 'FUS', 'TDP-43', 'hnRNPA1', 'hnRNPK',
                                 'DAPI_BIII_SFPQ', 'DAPI_BIII_FUS', 'DAPI_BIII_TDP-43', 'DAPI_BIII_hnRNPA1',
                                 'DAPI_BIII_hnRNPK']), required=True,
              help='''Choose one of the 13 experts (corresponds to a combination of channels)''')
@click.option('-la', '--label', type=click.Choice(['control', 'als']),
              required=True, help='''Label of images on which the expert will be evaluated''')
@click.option('-co', '--condition', type=click.Choice(['untreated', 'osmotic', 'oxidative', 'heat', 'osmotic_1h',
                                                       'osmotic_2h', 'osmotic_6h', 'heat_2h']),
              required=True, help='''Condition of images on which the expert will be evaluated''')
@click.option('--dry_run', is_flag=True, help="Dry run for testing")
def evaluate(config_name, classifier, expert, label, condition, dry_run=False):
    """Evaluate a model"""

    # get configuration parameters
    config = utils.get_config(config_name)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # initialize mobilenet pre-trained model
    model = models.mobilenet_v2(pretrained=True)
    for f in model.features:
        f.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model.to(device)

    # choose first fold of cross-validation (models were all trained on first fold)
    fold = 0

    # the name of the classifier varies with the type of classification
    if classifier == 'als':
        classifier_str = 'control_als_untreated'
        classification = 'als'
    else:
        classifier_str = f'control_untreated_{classifier}'
        classification = 'stress'

    # the chosen expert determines which channels will be used for training
    if expert == 'DAPI' or expert == 'BIII' or expert == 'DAPI_BIII':
        rbp = 'all'
        channels = expert.split('_')
    elif 'DAPI_BIII' in expert:
        rbp = expert[10:]
        channels = ['DAPI', 'BIII', rbp]
    else:
        rbp = expert
        channels = [expert]

    classifier_protocol = f'{classifier_str}_{rbp}'  # protocol used for training
    test_protocol = Protocol([label], [condition], rbp)  # protocol used for evaluation

    print(f'expert:{classifier_protocol}_{"_".join(channels)}\ndata:{test_protocol.name}')
    print('---------------------------------------------')

    # create dataloaders
    database = Database()
    test_indices, evaluation_protocol = get_evaluation_data(database, classifier_protocol, test_protocol, fold)  # test protocol becomes evaluation protocol because it might be changed in get evaluation data
    test_loader, test_dataset = utils.create_test_dataloader(config, database, test_indices, classification,
                                                             evaluation_protocol, channels, fold)

    # load state dict of trained model and put model in evaluation mode
    state_dict = f'state_dict_{classifier_protocol}_{"_".join(channels)}_fold_{fold}.pt'
    path = os.path.join(utils.get_project_root(), f'models/{classifier}_models/{state_dict}')
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    if not dry_run:
        # evaluate model with test loader iterating over test data and write results in csv
        results = pd.DataFrame(columns=['probabilities', 'indices'])
        _, y_targets, _, _, y_probabilities, y_indices = utils.validate(model, test_loader, device, save_metrics=True)
        results['probabilities'] = y_probabilities
        results['indices'] = y_indices
        write_results_to_file(classifier_str, channels, test_indices, evaluation_protocol, results)

    # delete folder with images
    test_dataset.delete()


def write_results_to_file(classifier_str: str, channels: List[str], test_indices: pd.Index,
                          evaluation_protocol: Protocol, results: pd.DataFrame, dry_run: bool = False):
    """Write mean probability for each image in csv file"""
    print(evaluation_protocol.name)
    path = os.path.join(utils.get_project_root(), f'results/image_probabilities.csv')
    file = pd.read_csv(path)
    channel = f'{"_".join(channels)}'

    if dry_run:  # rewrite already existing data - for test only
        probabilities = file[f'expert_{classifier_str}_{channel}'].dropna()
        for i, p in probabilities.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                file[f'expert_{classifier_str}_{channel}'].loc[i] = p

    else:
        probabilities = results.groupby('indices')['probabilities'].mean()  # compute the mean probability per image
        print(len(probabilities), ' images')
        path = os.path.join(utils.get_project_root(), f'data/protocols/{evaluation_protocol.name}.csv')
        images = pd.read_csv(path).loc[test_indices]  # find which images were evaluated
        images.reset_index(drop=True, inplace=True)

        # match image in csv file and assign probability
        for i, p in probabilities.items():
            image = images.loc[i]
            index = file.query(
                'experiment==@image.experiment and plate==@image.plate and neuron_type==@image.neuron_type and '
                'condition==@image.condition and stress_label==@image.stress_label and cell_line==@image.cell_line '
                'and als_label==@image.als_label and well_row==@image.well_row and well_col==@image.well_col '
                'and fov==@image.fov and number_of_planes==@image.number_of_planes and channel==@channel').index
            if len(index) == 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    file[f'expert_{classifier_str}_{channel}'].loc[index] = p
            elif len(index) > 1:
                print(i, ' - more than one image')
            elif len(index) == 0:
                print(i, ' - no image found')
    path = os.path.join(utils.get_project_root(), f'results/image_probabilities.csv')
    file.to_csv(path, index=False)


def get_evaluation_data(database: Database, classifier_protocol: str, protocol: Protocol, fold: int) -> (pd.Index, Protocol):
    """Return evaluation data based on which classifier is used - allows to avoid using data used during training"""
    # Note: this assumes that there is only one label or one condition in protocol ...
    if protocol.conditions[0] in classifier_protocol and protocol.labels[0] in classifier_protocol:  # some samples from this protocol were used during training
        train_protocol = Protocol.from_name(classifier_protocol)
        if 'control_als_untreated' in classifier_protocol:
            train, _ = database.cross_validation('als', train_protocol, fold)
            query = 'als_label==@protocol.labels[0]'
        else:
            train, _ = database.cross_validation('stress', train_protocol, fold)
            query = 'condition==@protocol.conditions[0]'
        samples = database.get_protocol_csv(train_protocol)
        samples.drop(index=pd.Index(train), inplace=True)  # drop corresponding samples used during training
        samples = samples.query(query)
        evaluation_protocol = train_protocol  # select samples from this protocol in dataset.py
    else:  # no sample was used during training
        samples = database.get_protocol_csv(protocol)
        evaluation_protocol = protocol
    return samples.index, evaluation_protocol


if __name__ == "__main__":
    evaluate()
