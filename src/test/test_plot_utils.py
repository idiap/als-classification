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

from src.visualization.visualize_utils import get_auc_results
import pytest
from src.protocol import RBPS


@pytest.mark.parametrize("label, condition",
                         [
                             ('control_als', 'untreated'),
                             ('control', 'untreated_osmotic'),
                             ('control', 'untreated_oxidative'),
                             ('control', 'untreated_heat'),
                             ('control', 'untreated_osmotic_1h'),
                             ('control', 'untreated_osmotic_2h'),
                             ('control', 'untreated_osmotic_6h'),
                             ('control', 'untreated_heat_2h'),
                         ]
                         )
def test_get_auc_results(label, condition):
    """Check that correct lines are extracted from result file """
    retval = get_auc_results(label, condition)
    for key, value in retval.items():
        if key in ['DAPI', 'BIII', 'DAPI_BIII']:
            for i, rbp in enumerate(RBPS[:-1]):
                assert value.index.values[i] == f'{label}_{condition}_{rbp}_{key}'
        elif key.startswith('DAPI_BIII'):
            rbp = key.split('_')[2]
            assert value.index.values[0] == f'{label}_{condition}_{rbp}_{key}'
        elif 'DAPI' not in key and 'BIII' not in key and 'DAPI_BIII' not in key:
            rbp = key.split('_')[0]
            assert value.index.values[0] == f'{label}_{condition}_{rbp}_{key}'
        else:
            assert value.index.values[0] == f'{label}_{condition}_{key}'
