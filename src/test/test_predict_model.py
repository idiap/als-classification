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

from src.predict_model import evaluate, get_evaluation_data
from src.database import Database
from src.protocol import Protocol
from click.testing import CliRunner
from src import utils
import pytest
import pandas as pd
import os

EXPERTS = ['DAPI',
           'BIII',
           'DAPI_BIII',
           'SFPQ',
           'TDP-43',
           'FUS',
           'hnRNPA1',
           'hnRNPK',
           'DAPI_BIII_SFPQ',
           'DAPI_BIII_TDP-43',
           'DAPI_BIII_FUS',
           'DAPI_BIII_hnRNPA1',
           'DAPI_BIII_hnRNPK']


def test_click_runner():
    runner = CliRunner()
    runner.invoke(evaluate, '-c user -cl als -e DAPI_BIII_hnRNPK -la control -co untreated --dry_run')


database = Database()
fold = 0


@pytest.mark.parametrize("classifier, protocol_name",
                         [
                             ('control_als_untreated_SFPQ', 'control_untreated_SFPQ'),
                             ('control_als_untreated_SFPQ', 'als_untreated_SFPQ'),
                             ('control_untreated_osmotic_SFPQ', 'control_untreated_SFPQ'),
                             ('control_untreated_osmotic_SFPQ', 'control_osmotic_SFPQ'),
                             ('control_untreated_heat_SFPQ', 'control_untreated_SFPQ'),
                             ('control_untreated_heat_SFPQ', 'control_heat_SFPQ'),
                             ('control_untreated_oxidative_SFPQ', 'control_untreated_SFPQ'),
                             ('control_untreated_oxidative_SFPQ', 'control_oxidative_SFPQ'),

                             ('control_als_untreated_FUS', 'control_untreated_FUS'),
                             ('control_als_untreated_FUS', 'als_untreated_FUS'),
                             ('control_untreated_osmotic_FUS', 'control_untreated_FUS'),
                             ('control_untreated_osmotic_FUS', 'control_osmotic_FUS'),
                             ('control_untreated_heat_FUS', 'control_untreated_FUS'),
                             ('control_untreated_heat_FUS', 'control_heat_FUS'),
                             ('control_untreated_oxidative_FUS', 'control_untreated_FUS'),
                             ('control_untreated_oxidative_FUS', 'control_oxidative_FUS'),

                             ('control_als_untreated_TDP-43', 'control_untreated_TDP-43'),
                             ('control_als_untreated_TDP-43', 'als_untreated_TDP-43'),
                             ('control_untreated_osmotic_TDP-43', 'control_untreated_TDP-43'),
                             ('control_untreated_osmotic_TDP-43', 'control_osmotic_TDP-43'),
                             ('control_untreated_heat_TDP-43', 'control_untreated_TDP-43'),
                             ('control_untreated_heat_TDP-43', 'control_heat_TDP-43'),
                             ('control_untreated_oxidative_TDP-43', 'control_untreated_TDP-43'),
                             ('control_untreated_oxidative_TDP-43', 'control_oxidative_TDP-43'),

                             ('control_als_untreated_hnRNPA1', 'control_untreated_hnRNPA1'),
                             ('control_als_untreated_hnRNPA1', 'als_untreated_hnRNPA1'),
                             ('control_untreated_osmotic_hnRNPA1', 'control_untreated_hnRNPA1'),
                             ('control_untreated_osmotic_hnRNPA1', 'control_osmotic_hnRNPA1'),
                             ('control_untreated_heat_hnRNPA1', 'control_untreated_hnRNPA1'),
                             ('control_untreated_heat_hnRNPA1', 'control_heat_hnRNPA1'),
                             ('control_untreated_oxidative_hnRNPA1', 'control_untreated_hnRNPA1'),
                             ('control_untreated_oxidative_hnRNPA1', 'control_oxidative_hnRNPA1'),

                             ('control_als_untreated_hnRNPK', 'control_untreated_hnRNPK'),
                             ('control_als_untreated_hnRNPK', 'als_untreated_hnRNPK'),
                             ('control_untreated_osmotic_hnRNPK', 'control_untreated_hnRNPK'),
                             ('control_untreated_osmotic_hnRNPK', 'control_osmotic_hnRNPK'),
                             ('control_untreated_heat_hnRNPK', 'control_untreated_hnRNPK'),
                             ('control_untreated_heat_hnRNPK', 'control_heat_hnRNPK'),
                             ('control_untreated_oxidative_hnRNPK', 'control_untreated_hnRNPK'),
                             ('control_untreated_oxidative_hnRNPK', 'control_oxidative_hnRNPK'),

                             ('control_als_untreated_hnRNPK', 'control_untreated_hnRNPK'),
                             ('control_als_untreated_hnRNPK', 'als_untreated_hnRNPK'),
                             ('control_untreated_osmotic_hnRNPK', 'control_untreated_hnRNPK'),
                             ('control_untreated_osmotic_hnRNPK', 'control_osmotic_hnRNPK'),
                             ('control_untreated_heat_hnRNPK', 'control_untreated_hnRNPK'),
                             ('control_untreated_heat_hnRNPK', 'control_heat_hnRNPK'),
                             ('control_untreated_oxidative_hnRNPK', 'control_untreated_hnRNPK'),
                             ('control_untreated_oxidative_hnRNPK', 'control_oxidative_hnRNPK'),

                             ('control_als_untreated_all', 'control_untreated_all'),
                             ('control_als_untreated_all', 'als_untreated_all'),
                             ('control_untreated_osmotic_all', 'control_untreated_all'),
                             ('control_untreated_osmotic_all', 'control_osmotic_all'),
                             ('control_untreated_heat_all', 'control_untreated_all'),
                             ('control_untreated_heat_all', 'control_heat_all'),
                             ('control_untreated_oxidative_all', 'control_untreated_all'),
                             ('control_untreated_oxidative_all', 'control_oxidative_all'),
                         ])
def test_get_evaluation_data_with_training_data(classifier, protocol_name):
    protocol = Protocol.from_name(protocol_name)
    samples = database.get_protocol_data(protocol)
    data, evaluation_protocol = get_evaluation_data(database, classifier, protocol, fold)
    train_protocol = Protocol.from_name(classifier)
    assert evaluation_protocol.name == train_protocol.name
    if 'control_als_untreated' in classifier:
        train, _ = database.cross_validation('als', train_protocol, fold)
        query = 'als_label==@protocol.labels[0]'
    else:
        train, _ = database.cross_validation('stress', train_protocol, fold)
        query = 'condition==@protocol.conditions[0]'
    path = os.path.join(utils.get_project_root(), f'data/protocols/{evaluation_protocol.name}.csv')
    train_samples = pd.read_csv(path).loc[train]
    # assert that evaluation data is equal to samples minus those needed for training
    assert len(samples) - len(train_samples.query(query)) == len(data)


def test_correct_channels_in_probabilities():
    """Check that classifier was only evaluated with images with specified channel(s)"""
    directory = os.path.join(utils.get_project_root(), 'results')
    df = pd.read_csv(f'{directory}/image_probabilities.csv')
    for classifier in ['control_als_untreated', 'control_untreated_osmotic', 'control_untreated_heat',
                       'control_untreated_oxidative']:
        for expert in EXPERTS:
            index = df[f'expert_{classifier}_{expert}'].dropna().index
            assert (df['channel'].loc[index] == expert).all()
