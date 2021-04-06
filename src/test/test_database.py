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
import pandas as pd
import os
from src import utils
import pytest

db = Database()
path = os.path.join(utils.get_project_root(), 'data/protocols/protocols_stats.csv')
protocol_stats = pd.read_csv(path, index_col=0)


@pytest.mark.parametrize("protocol_name,images, samples, als, control, stress, no_stress",
                         protocol_stats.to_records(index=True)
                         )
def test_get_protocol_data(protocol_name, images, samples, als, control, stress, no_stress):
    protocol = Protocol.from_name(protocol_name)
    data = db.get_protocol_data(protocol)

    assert images == data['number_of_planes'].sum() * 3
    assert samples == len(data)
    assert als == (data.als_label == 'als').sum()
    assert control == (data.als_label == 'control').sum()
    assert stress == (data.stress_label == 'stress').sum()
    assert no_stress == (data.stress_label == 'no_stress').sum()


@pytest.mark.parametrize("labels, conditions, rbp, expected_results",
                         [
                             (['control'], ['untreated', 'osmotic'], 'SFPQ', (464, 120)),
                             (['control'], ['untreated', 'oxidative'], 'SFPQ', (392, 48)),
                             (['control'], ['untreated', 'heat'], 'SFPQ', (488, 144)),
                             (['control'], ['untreated', 'osmotic'], 'all', (1909, 480)),
                             (['control'], ['untreated', 'oxidative'], 'all', (1693, 264)),
                             (['control'], ['untreated', 'heat'], 'all', (1861, 432)),
                             (['control'], ['untreated', 'osmotic_1h'], 'SFPQ', (464, 120)),
                             (['control'], ['untreated', 'heat_2h'], 'SFPQ', (488, 144)),
                             (['control'], ['untreated', 'osmotic_2h'], 'SFPQ', (464, 120)),
                             (['control'], ['untreated', 'osmotic_6h'], 'SFPQ', (416, 72)),
                             (['control'], ['untreated', 'osmotic_1h'], 'all', (1789, 360)),
                             (['control'], ['untreated', 'osmotic_2h'], 'all', (1789, 360)),
                             (['control'], ['untreated', 'osmotic_6h'], 'all', (1645, 216)),
                             (['control'], ['untreated', 'heat_2h'], 'all', (1861, 432)),
                             (['control', 'als'], ['untreated'], 'SFPQ', (848, 344)),
                             (['control', 'als'], ['osmotic'], 'SFPQ', (312, 120)),
                             (['control', 'als'], ['oxidative'], 'SFPQ', (143, 48)),
                             (['control', 'als'], ['heat'], 'SFPQ', (336, 144)),
                             (['control', 'als'], ['untreated'], 'all', (3419, 1429)),
                             (['control', 'als'], ['osmotic'], 'all', (1216, 480)),
                             (['control', 'als'], ['oxidative'], 'all', (711, 264)),
                             (['control', 'als'], ['heat'], 'all', (1008, 432))
                         ]
                         )
def test_limit_samples(labels, conditions, rbp, expected_results):
    protocol = Protocol(labels, conditions, rbp)
    assert db.get_limit_samples(protocol) == expected_results


@pytest.mark.parametrize("classification, protocol_name",
                         # als training protocols
                         [('als', 'control_als_untreated_SFPQ'),
                          ('als', 'control_als_untreated_TDP-43'),
                          ('als', 'control_als_untreated_FUS'),
                          ('als', 'control_als_untreated_hnRNPA1'),
                          ('als', 'control_als_untreated_hnRNPK'),
                          ('als', 'control_als_untreated_all'),
                          # osmotic training protocols
                          ('stress', 'control_untreated_osmotic_SFPQ'),
                          ('stress', 'control_untreated_osmotic_TDP-43'),
                          ('stress', 'control_untreated_osmotic_FUS'),
                          ('stress', 'control_untreated_osmotic_hnRNPA1'),
                          ('stress', 'control_untreated_osmotic_hnRNPK'),
                          ('stress', 'control_untreated_osmotic_all'),
                          ('stress', 'control_untreated_osmotic_1h_SFPQ'),
                          ('stress', 'control_untreated_osmotic_1h_TDP-43'),
                          ('stress', 'control_untreated_osmotic_1h_FUS'),
                          ('stress', 'control_untreated_osmotic_1h_hnRNPA1'),
                          ('stress', 'control_untreated_osmotic_1h_hnRNPK'),
                          ('stress', 'control_untreated_osmotic_1h_all'),
                          ('stress', 'control_untreated_osmotic_2h_SFPQ'),
                          ('stress', 'control_untreated_osmotic_2h_TDP-43'),
                          ('stress', 'control_untreated_osmotic_2h_FUS'),
                          ('stress', 'control_untreated_osmotic_2h_hnRNPA1'),
                          ('stress', 'control_untreated_osmotic_2h_hnRNPK'),
                          ('stress', 'control_untreated_osmotic_2h_all'),
                          ('stress', 'control_untreated_osmotic_6h_SFPQ'),
                          ('stress', 'control_untreated_osmotic_6h_TDP-43'),
                          ('stress', 'control_untreated_osmotic_6h_FUS'),
                          ('stress', 'control_untreated_osmotic_6h_hnRNPA1'),
                          ('stress', 'control_untreated_osmotic_6h_hnRNPK'),
                          ('stress', 'control_untreated_osmotic_6h_all'),
                          # heat training protocols
                          ('stress', 'control_untreated_heat_SFPQ'),
                          ('stress', 'control_untreated_heat_TDP-43'),
                          ('stress', 'control_untreated_heat_FUS'),
                          ('stress', 'control_untreated_heat_hnRNPA1'),
                          ('stress', 'control_untreated_heat_hnRNPK'),
                          ('stress', 'control_untreated_heat_all'),
                          ('stress', 'control_untreated_heat_2h_SFPQ'),
                          ('stress', 'control_untreated_heat_2h_TDP-43'),
                          ('stress', 'control_untreated_heat_2h_FUS'),
                          ('stress', 'control_untreated_heat_2h_hnRNPA1'),
                          ('stress', 'control_untreated_heat_2h_hnRNPK'),
                          ('stress', 'control_untreated_heat_2h_all'),
                          # oxidative training protocols
                          ('stress', 'control_untreated_oxidative_SFPQ'),
                          ('stress', 'control_untreated_oxidative_TDP-43'),
                          ('stress', 'control_untreated_oxidative_FUS'),
                          ('stress', 'control_untreated_oxidative_hnRNPA1'),
                          ('stress', 'control_untreated_oxidative_hnRNPK'),
                          ('stress', 'control_untreated_oxidative_all')

                          ]
                         )
def test_cross_validation(classification, protocol_name):
    protocol = Protocol.from_name(protocol_name)
    data = db.get_protocol_data(protocol, save=False)
    if classification == 'stress':
        label = 'stress_label'
        pos = 'stress'
        neg = 'no_stress'
    elif classification == 'als':
        label = 'als_label'
        pos = 'als'
        neg = 'control'
    for fold in range(10):
        train, test = db.cross_validation(classification, protocol, fold)
        assert not bool(set(train) & set(test))  # assert no intersection between train and test set
        train_label_cnt = data.loc[train, label].value_counts()
        test_label_cnt = data.loc[test, label].value_counts()
        pos_train = train_label_cnt[pos]
        neg_train = train_label_cnt[neg]
        pos_test = test_label_cnt[pos]
        neg_test = test_label_cnt[neg]

        assert pos_test + pos_train == neg_test + neg_train  # assert that classes are balanced
