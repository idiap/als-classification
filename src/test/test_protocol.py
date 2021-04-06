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

from src.protocol import Protocol
import pytest


@pytest.mark.parametrize("labels,conditions,rbp",
                         [(['als', 'control', 'control'], ['untreated', 'osmotic'], 'TDP-43'),
                          (['als', 'control'], ['untreated', 'osmotic', 'oxidative'], 'SFPQ'),
                          (['als', 'control'], ['untreated'], 'ABC'),
                          (['ctrl'], ['heat'], 'hnRNPK'),
                          (['als'], ['cellular'], 'TDP-43')
                          ]
                         )
def test_wrong_protocols(labels, conditions, rbp):
    try:
        Protocol(labels, conditions, rbp)
        pytest.fail()
    except AssertionError as e:
        assert 'Wrong protocol initialization' in e.args[0]


def test_correct_protocols():
    protocol = Protocol(['als', 'control'], ['untreated'], 'SFPQ')
    assert protocol.name == 'control_als_untreated_SFPQ'


def test_protocol_names_sorted():
    protocol_1 = Protocol(['als', 'control'], ['untreated', 'osmotic'], 'TDP-43')
    protocol_2 = Protocol(['control', 'als'], ['osmotic', 'untreated'], 'TDP-43')
    assert protocol_1.name == protocol_2.name


@pytest.mark.parametrize("name",
                         ['control_als_osmotic_hnRNPK',
                          'als_untreated_oxidative_TDP-43',
                          'als_untreated_osmotic_6h_SFPQ',
                          'control_heat_2h_hnRNPA1'
                          ])
def test_protocol_from_name(name):
    assert Protocol.from_name(name).name == name


def test_protocol_from_name_sorted():
    assert Protocol.from_name('als_oxidative_untreated_TDP-43').name == 'als_untreated_oxidative_TDP-43'
