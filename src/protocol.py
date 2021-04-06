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

import re
from typing import List

LABELS = ['control', 'als']
CONDITIONS = ['untreated', 'oxidative', 'heat', 'heat_2h', 'osmotic', 'osmotic_1h', 'osmotic_2h', 'osmotic_6h']
RBPS = ['SFPQ', 'TDP-43', 'FUS', 'hnRNPA1', 'hnRNPK', 'all']


class Protocol:
    """Define some rules to select samples in the database"""

    def __init__(self, labels: List[str], conditions: List[str], rbp: str):
        """Initialize a protocol

        Args:
            labels (list[str]): list of 1 or 2 labels from LABELS
            conditions (list[str]) : list of of 1 or 2 conditions from CONDITIONS
            rbp (str): RNA-binding protein from RBPS
        """
        self.check_initialization(labels, conditions, rbp)
        self.labels = sorted(labels,
                             reverse=True)  # sort labels in reverse alphabetical order so that the name is always the same
        self.conditions = sorted(conditions,
                                 reverse=True)  # sort conditions in reverse alphabetical order so that the name is always the same
        self.rbp = rbp

    def __str__(self):
        """Write fields of the protocol"""
        return f'labels: {self.labels}\nconditions: {self.conditions}\nrbp: {self.rbp}'

    def set_labels(self, labels: List[str]):
        """Set new labels"""
        self.labels = labels

    def set_conditions(self, conditions: List[str]):
        """Set new conditions"""
        self.conditions = conditions

    def set_rbp(self, rbp: str):
        """Set new rbp"""
        self.rbp = rbp

    @staticmethod
    def check_initialization(labels: List[str], conditions: List[str], rbp: str):
        """Verify that arguments for initialization are valid"""

        assert len(labels) in [1, 2], 'Wrong protocol initialization - Choose either one or two labels'
        assert len(conditions) in [1, 2], 'Wrong protocol initialization - Choose either one or two conditions'
        assert rbp in RBPS, f'Wrong protocol initialization - "{rbp}" is not is the list of rbps'
        for label in labels:
            assert label in LABELS, f'Wrong protocol initialization - "{label}" is not is the list of labels'
        for condition in conditions:
            assert condition in CONDITIONS, f'Wrong protocol initialization - "{condition}" is not is the list of conditions'

    @property
    def name(self):
        """Return name of the protocol as follows: label_conditions_rbp"""
        return f'{"_".join(self.labels)}_{"_".join(self.conditions)}_{self.rbp}'

    @classmethod
    def from_name(cls, name: str):
        """Create protocol from its name"""
        labels, conditions, rbp = re.findall(r'^(control_als|als|control)_([a-z_\d]+)_(.+)$', name)[
            0]  # use regex to find the labels, conditions and rbp in the name
        labels = labels.split('_')
        if '_' in conditions and 'untreated' in conditions:  # if there are several conditions
            conditions = conditions.split('_', 1)
        else:
            conditions = [conditions]  # put condition in a list
        return cls(labels, conditions, rbp)
