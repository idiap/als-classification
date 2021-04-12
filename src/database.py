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

from src.protocol import Protocol, RBPS
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import utils
import os


class Database:
    """Define the fluoMNs database based on the annotation file"""
    def __init__(self):
        """Initialize the database"""
        # initialize the annotation file to be able to open the images
        path = os.path.join(utils.get_project_root(), 'data/', 'annotations_fluoMNs.csv')
        self.annotations = pd.read_csv(path)
        # initialize all valid samples - some are excluded due to bad experimental conditions
        self.all_samples = self.annotations.query('exclude=="no"')
        # initialize all valid protocols associated to the database
        self.protocols = []
        list_labels = [['als'], ['control'], ['control', 'als']]
        list_conditions = [['untreated'], ['osmotic'], ['oxidative'], ['heat'],
                           ['untreated', 'osmotic'], ['untreated', 'oxidative'], ['untreated', 'heat'],
                           ['untreated', 'heat_2h'], ['untreated', 'osmotic_1h'], ['untreated', 'osmotic_2h'],
                           ['untreated', 'osmotic_6h'], ['osmotic_1h'], ['osmotic_2h'], ['osmotic_6h'], ['heat_2h']]
        channels = ['SFPQ', 'TDP-43', 'FUS', 'hnRNPA1', 'hnRNPK', 'all']
        for labels in list_labels:
            for conditions in list_conditions:
                for channel in channels:
                    self.protocols.append(Protocol(labels, conditions, channel))

    def get_protocol_data(self, protocol: Protocol, save: bool = False) -> pd.DataFrame:
        """Return dataframe with samples specified by the protocol"""
        data = self.all_samples
        # create masks to filter the dataframe
        neuron_d6 = data.neuron_type == "D6"
        labels = data.als_label.isin(protocol.labels)
        conditions = data.condition.isin(protocol.conditions)
        channels = (data.channel_1 == 'DAPI') & (data.channel_2 == 'BIII')
        rbp_filter = RBPS if protocol.rbp == 'all' else [protocol.rbp]
        rbp_channels = (data.channel_3.isin(rbp_filter)) | (data.channel_4.isin(rbp_filter))

        data = data[neuron_d6 & labels & conditions & channels & rbp_channels]
        data = data.reset_index(drop=True)  # dataframe index is reset starting from 0

        if save:  # save the dataframe in a csv file
            path = os.path.join(utils.get_project_root(), 'data/protocols', f'{protocol.name}.csv')
            data.to_csv(path, index=False)
        return data

    def get_protocol_csv(self, protocol: Protocol) -> pd.DataFrame:
        """Create protocol csv file (if not already created) and return dataframe"""
        path = os.path.join(utils.get_project_root(), f'data/protocols/{protocol.name}.csv')
        if os.path.exists(path):
            samples = pd.read_csv(path)
        else:
            samples = self.get_protocol_data(protocol, save=True)
        return samples

    def create_all_protocols(self):
        """Create and save a csv file for each protocol"""
        for protocol in self.protocols:
            self.get_protocol_data(protocol, save=True)

    def cross_validation(self, classification: str, protocol: Protocol, fold: int) -> (list, list):
        """Return train indices and test indices for the given fold"""
        samples = self.get_protocol_csv(protocol)
        label = f'{classification}_label'
        # adjust to have a limited number of samples to allow comparisons between protocols:
        limit_samples, min_samples = self.get_limit_samples(protocol)
        if min_samples < limit_samples/2:
            samples = samples.groupby(label).sample(n=min_samples, random_state=42)  # balance the two classes
        else:
            samples = samples.groupby(label).sample(n=int(limit_samples/2), random_state=42)  # balance the two classes

        x = samples.drop(label, axis=1).reset_index()  # puts original index in a column
        y = samples[label].reset_index(drop=True)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        train_indices, test_indices = dict(), dict()
        for i, (train_index, test_index) in enumerate(skf.split(x, y)):
            train_indices[i] = x.loc[train_index, 'index'].values  # get original indices
            test_indices[i] = x.loc[test_index, 'index'].values
        return train_indices[fold], test_indices[fold]

    def get_limit_samples(self, protocol: Protocol) -> (int, int):
        """Return the maximum number of samples and the minimum samples of each class"""
        original_rbp = protocol.rbp  # keep it to reassign it at the end of function
        try:
            path = os.path.join(utils.get_project_root(), 'data/protocols/protocols_stats.csv')
            data = pd.read_csv(path, index_col='protocol')
        except FileNotFoundError:
            self.compute_protocols_stats(debug=False)
            data = pd.read_csv(path, index_col='protocol')

        if protocol.labels == ['control', 'als']:
            classes = protocol.labels
        else:
            classes = ['stress', 'no_stress']

        if protocol.rbp == 'all':
            data = data.loc[protocol.name]
            return data['samples'], data[classes].min()
        else:
            # take minimum over all RBPs
            protocols = []
            for rbp in RBPS:
                protocol.set_rbp(rbp)
                protocols.append(protocol.name)

            data = data.loc[protocols]
            protocol.set_rbp(original_rbp)  # reassign original rbp
            return data['samples'].min(), (data[classes].min()).min()

    def compute_protocols_stats(self, debug: bool = False):
        """Create statistics about protocols - needed for get_limit_samples"""
        protocol_names = [protocol.name for protocol in self.protocols]
        protocols_stats = pd.DataFrame(columns=['images', 'samples', 'als', 'control', 'stress', 'no_stress'],
                                       index=protocol_names)
        for protocol in self.protocols:
            samples = self.get_protocol_data(protocol)
            protocols_stats['samples'].loc[protocol.name] = len(samples)
            images = 0
            for i, sample in samples.iterrows():
                images += sample.number_of_planes * 3  # 3 channels: DAPI, BIII and RBP (protocol.rbp)
            protocols_stats['images'].loc[protocol.name] = images
            for als_label in ['als', 'control']:
                try:
                    protocols_stats[als_label].loc[protocol.name] = samples['als_label'].value_counts()[als_label]
                except KeyError:
                    protocols_stats[als_label].loc[protocol.name] = 0
            for stress_label in ['stress', 'no_stress']:
                try:
                    protocols_stats[stress_label].loc[protocol.name] = samples['stress_label'].value_counts()[stress_label]
                except KeyError:
                    protocols_stats[stress_label].loc[protocol.name] = 0
            if debug:
                print(protocols_stats.loc[protocol.name])
        path = os.path.join(utils.get_project_root(), 'data/protocols/protocols_stats.csv')
        protocols_stats.to_csv(path, index_label='protocol')
