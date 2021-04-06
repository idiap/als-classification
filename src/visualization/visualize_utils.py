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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kstest
from src import utils
from src.database import Database
from src.protocol import RBPS, LABELS, CONDITIONS
import math
import os
import matplotlib as mpl
from typing import Dict, List

mpl.rcParams['pdf.fonttype'] = 42

acronyms = {
    'untreated': 'UT',
    'osmotic': 'OSM',
    'oxidative': 'OX',
    'heat': 'HS',
    'osmotic_1h': '1h',
    'osmotic_2h': '2h',
    'osmotic_6h': '6h',
    'heat_2h': '2h',
    'control': 'CTRL',
    'als': 'ALS'
}

marker_colors = {
    'SFPQ': 'tab:orange',
    'FUS': 'tab:green',
    'TDP-43': 'tab:red',
    'hnRNPA1': 'tab:purple',
    'hnRNPK': 'tab:cyan',
    'DAPI': 'b',
    'BIII': 'r',
    'DAPI_BIII': 'm'
}

condition_colors = {
    'untreated': '#989898',
    'osmotic': '#E4BB21',
    'oxidative': '#7B2A8F',
    'heat': '#377092',
    'heat_2h': '#DAE5EC',
    'osmotic_1h': '#ECD78B',
    'osmotic_2h': '#F2E1A9',
    'osmotic_6h': '#F8EECE',
}

directory = os.path.join(utils.get_project_root(), "reports/figures/")


def cm_to_inch(value):
    return value / 2.54


def figure_1D():
    """Distribution of images in the dataset"""
    print('FIGURE 1D')
    fig = plot_number_of_images(figsize=(7, 9))
    fig.savefig(f'{directory}/Figure_1D.pdf', dpi=300)


def figure_2A():
    """Boxplots of AUC of DAPI, BIII and DAPI-BIII als-experts """
    print('FIGURE 2A')
    fig = plot_auc_results('als', ['DAPI', 'BIII', 'DAPI_BIII'], figsize=(6, 5.5))
    fig.savefig(f'{directory}/Figure_2A.pdf', dpi=300)
    print('FIGURE S2A')
    fig = plot_auc_results('als', ['DAPI', 'BIII', 'DAPI_BIII'], figsize=(6, 6), scatter=True)
    fig.savefig(f'{directory}/Figure_S2A.pdf')


def figure_2C():
    """Violin plots of probabilities for CTRL stress images to be labeled ALS by DAPI, BIII and DAPI-BIII als-experts"""
    print('FIGURE 2C')
    fig = plot_probabilities_per_expert('control_als_untreated', 'control',
                                        ['untreated', 'oxidative', 'osmotic', 'heat'],
                                        ['DAPI', 'BIII', 'DAPI_BIII'], figsize=(15.5, 6))
    fig.savefig(f'{directory}/Figure_2C.pdf', dpi=300)


def figure_2D():
    """Boxplots of AUC of DAPI, BIII and DAPI-BIII stress- and stress-recovery- experts """
    print('FIGURE 2D and S2D')
    for stress in ['oxidative_stress', 'osmotic_stress', 'heat_stress', 'heat_stress_2h', 'osmotic_stress_1h',
                   'osmotic_stress_2h', 'osmotic_stress_6h', ]:
        fig = plot_auc_results(stress, ['DAPI', 'BIII', 'DAPI_BIII'], figsize=(6, 5.5))
        fig.savefig(f'{directory}/Figure_2D_{stress.replace("_stress", "")}.pdf', dpi=300)
        fig = plot_auc_results(stress, ['DAPI', 'BIII', 'DAPI_BIII'], figsize=(6, 6), scatter=True)
        fig.savefig(f'{directory}/Figure_S2D_{stress.replace("_stress", "")}.pdf', dpi=300)


def figure_3A():
    """Boxplots of AUC of DAPI and RBP als-experts """
    print('FIGURE 3A')
    fig = plot_auc_results('als', ['DAPI', 'RBP'], figsize=(9.5, 7), sorted=True)
    fig.savefig(f'{directory}/Figure_3A.pdf', dpi=300)


def figure_3C():
    """Boxplots of AUC of DAPI-BIII and DAPI-BIII-RBP als-experts + barplot of log10 p-values"""
    print('FIGURE 3C and S3C')
    fig, fig_pvalue = plot_auc_results('als', ['DAPI_BIII', 'DAPI_BIII_RBP'], figsize=(15, 7),
                                       sorted=True, plot_pvalue=True)
    fig_pvalue.savefig(f'{directory}/Figure_3C.pdf', dpi=300)
    fig.savefig(f'{directory}/Figure_S3C.pdf', dpi=300)


def figure_3D():
    """Violin plots of probabilities for CTRL stress images to be labeled ALS by DAPI-BIII and DAPI-BIII-RBP als-experts"""
    print('FIGURE 3D')
    fig = plot_probabilities_per_condition('control_als_untreated', 'control',
                                           ['oxidative', 'heat', 'osmotic'],
                                           [f'DAPI_BIII_{rbp}' for rbp in RBPS[:-1]], figsize=(21, 7))
    fig.savefig(f'{directory}/Figure_3D.pdf', dpi=300)


def figure_4A():
    """Boxplots of AUC of DAPI-BIII and DAPI-BIII-RBP stress-experts + barplot of log10 p-values"""
    print('FIGURE 4A')
    for stress in ['oxidative', 'heat', 'osmotic']:
        fig, fig_p_value = plot_auc_results(stress + '_stress', ['DAPI_BIII', 'DAPI_BIII_RBP'],
                                            figsize=(15, 7), sorted=True, plot_pvalue=True)
        fig_p_value.savefig(f'{directory}/Figure_4A_{stress}.pdf', dpi=300)
        fig.savefig(f'{directory}/Figure_S4A_{stress}.pdf', dpi=300)


def figure_4B():
    """Violin plots of probabilities for stress recovery images to be labeled STRESSED by DAPI-BIII-RBP stress-experts"""
    print('FIGURE 4B')
    for stress in ['heat', 'osmotic']:
        fig = plot_probabilities_stress_recovery(stress, 'control', 'DAPI_BIII_RBP',
                                                 figsize=(20, 6))
        fig.savefig(f'{directory}/Figure_4B_{stress}.pdf', dpi=300)


def figure_5A():
    """Boxplots of AUC of DAPI and RBP stress-experts + barplot of log10 p-values"""
    print('FIGURE 5A')
    for stress in ['oxidative', 'heat', 'osmotic']:
        fig, fig_p_value = plot_auc_results(stress + '_stress', ['DAPI', 'RBP'],
                                            figsize=(15, 8), sorted=True, plot_pvalue=True)
        fig_p_value.savefig(f'{directory}/Figure_5A_{stress}.pdf', dpi=300)
        fig.savefig(f'{directory}/Figure_S5A_{stress}.pdf', dpi=300)


def figure_5B():
    """Violin plots of probabilities for stress recovery images to be labeled STRESSED by RBP stress-experts"""
    print('FIGURE 5B')
    for stress in ['heat', 'osmotic']:
        fig = plot_probabilities_stress_recovery(stress, 'control', 'RBP',
                                                 figsize=(20, 6))
        fig.savefig(f'{directory}/Figure_5B_{stress}.pdf', dpi=300)


def figure_6A():
    """Boxplots of AUC of DAPI and RBP stress-experts """
    print('FIGURE 6A')
    for stress in ['oxidative', 'heat', 'osmotic']:
        fig = plot_auc_results(stress + '_stress', ['DAPI', 'RBP'], figsize=(9.5, 7), sorted=True)
        fig.savefig(f'{directory}/Figure_6A_{stress}.pdf', dpi=300)


# PLOT UTILS
# ----------------------------------------------------------------------------------------------------------------------
def get_number_images(data: pd.DataFrame) -> int:
    """Compute number of images in a given dataframe with samples"""
    images = 0
    for i, sample in data.iterrows():
        if sample.channel_3 in RBPS and sample.channel_4 in RBPS:
            images += 4 * sample.number_of_planes
        else:
            images += 3 * sample.number_of_planes
    return images


def plot_number_of_images(figsize: (int, int)) -> plt.figure:
    """Bar plot representing the repartition of images in the dataset"""
    db = Database()
    data = db.all_samples
    neuron_d6 = data.neuron_type == "D6"
    labels = data.als_label.isin(LABELS)
    conditions = data.condition.isin(CONDITIONS)
    channels = (data.channel_1 == 'DAPI') & (data.channel_2 == 'BIII')
    rbp_channels = (data.channel_3.isin(RBPS)) | (data.channel_4.isin(RBPS))
    data = data[neuron_d6 & labels & conditions & channels & rbp_channels]
    data = data.reset_index(drop=True)  # dataframe index is reset starting from 0
    images = dict()
    for label in LABELS:
        images[label] = dict()
        for condition in CONDITIONS:
            images[label][condition] = get_number_images(data.query('als_label==@label and condition==@condition'))
    print(f'Total number of images: {sum(images["control"].values()) + sum(images["als"].values())}')

    figsize = (cm_to_inch(figsize[0]), cm_to_inch(figsize[1]))
    fontsize = 8

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    for i, (label, image) in enumerate(images.items()):
        labels = []
        for j, (condition, number_images) in enumerate(image.items()):
            ax[i].bar(j, number_images, color=condition_colors[condition])
            labels.append(acronyms[condition])
        ax[i].set_xticks(range(8))
        ax[i].set_xticklabels(labels, fontsize=fontsize)
        ax[i].set_ylim([0, 35000])
        ax[i].yaxis.set_tick_params(labelsize=fontsize)
        ax[i].set_ylabel('Number of images', fontsize=fontsize)
        ax[i].set_title(f'{acronyms[label]} images', fontsize=fontsize)
    plt.tight_layout()
    plt.show()
    return fig


def get_auc_results(label: str, condition: str) -> Dict:
    """Collect AUC values for a given combination of labels and conditions"""
    results = pd.read_csv(f'{os.path.join(utils.get_project_root(), "results")}/auc.csv', index_col=0)
    data = results[results.index.str.contains(condition)]
    data = data[data.index.str.contains(label)]
    data = results[results.index.str.contains(f'{label}_{condition}')]
    if condition == 'untreated_heat':
        data = data[~data.index.str.contains(f'{label}_{condition}_2h')]
    elif condition == 'untreated_osmotic':
        data = data[~data.index.str.contains(f'{label}_{condition}_1h')]
        data = data[~data.index.str.contains(f'{label}_{condition}_2h')]
        data = data[~data.index.str.contains(f'{label}_{condition}_6h')]
    data = data[~data.index.str.contains(f'{label}_{condition}_all')]
    retval = dict()
    for i, rbp in enumerate(RBPS[:-1]):
        rbp_data = data[data.index.str.contains(f'{condition}_{rbp}')]
        retval[f'{rbp}'] = rbp_data[rbp_data.index.str.contains(f'{rbp}_{rbp}')].dropna(axis=1)
        retval[f'{rbp}_DAPI'] = rbp_data[rbp_data.index.str.contains(r'_DAPI$')].dropna(axis=1)
        retval[f'{rbp}_BIII'] = rbp_data[rbp_data.index.str.contains(r'[^I]_BIII$')].dropna(axis=1)
        retval[f'{rbp}_DAPI_BIII'] = rbp_data[rbp_data.index.str.contains(r'DAPI_BIII$')].dropna(axis=1)
        retval[f'DAPI_BIII_{rbp}'] = rbp_data[rbp_data.index.str.contains(f'{rbp}_DAPI_BIII_{rbp}')].dropna(axis=1)
    retval['DAPI'] = pd.concat([retval[f'{rbp}_DAPI'] for rbp in RBPS[:-1]])
    retval['BIII'] = pd.concat([retval[f'{rbp}_BIII'] for rbp in RBPS[:-1]])
    retval['DAPI_BIII'] = pd.concat([retval[f'{rbp}_DAPI_BIII'] for rbp in RBPS[:-1]])
    return retval


def mannwhitney_ttest(ref: tuple, samples: Dict, alternative: str = 'two-sided') -> float:
    """Execute Mann-Whitney test comparing each sample with ref and print resulting p-values"""
    print(f'Mann-Whitney ({alternative})')
    for name, sample in samples.items():
        p_value = mannwhitneyu(ref[1], sample, alternative=alternative)[1]
        print(f'from {ref[0]} to {name}: {p_value}')
        return p_value


def kolmogorovsmirnov_ttest(ref: tuple, samples, alternative='two-sided'):
    """Execute Kolmogorov-Smirnov test comparing each sample with ref and print resulting p-values"""
    print(f'Kolmogorov-Smirnov ({alternative})')
    for name, sample in samples.items():
        p_value = kstest(ref[1], sample, alternative=alternative)[1]
        print(f'from {ref[0]} to {name}: {p_value}')
        return p_value


def plot_auc_results(classification: str, experts: List[str], figsize: (int, int), sorted: bool = False,
                     scatter: bool = False, plot_pvalue: bool = False) -> plt.figure:
    """Boxplots of AUC performances"""
    figsize = (cm_to_inch(figsize[0]), cm_to_inch(figsize[1]))
    fontsize = 8
    linewidth = 0.5
    marker = "."
    fig, ax = plt.subplots(figsize=figsize)

    if 'stress' in classification:
        stress = classification.replace('_stress', '')
        condition = f'untreated_{stress}'
        cell_line = 'control'
        title = stress
        ax.set_ylabel('AUC (untreated vs stress)', fontsize=fontsize)
        positions_p_value = [0.75, 0.65, 0.75]
    elif classification == 'als':
        condition = 'untreated'
        cell_line = 'control_als'
        title = 'als'
        ax.set_ylabel('AUC (ALS vs CTRL)', fontsize=fontsize)
        positions_p_value = [0.9, 1, 0.9]

    auc_results = get_auc_results(cell_line, condition)
    data_expert = dict()
    print(title)
    ax.set_title(title, fontsize=fontsize)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.yaxis.set_tick_params(labelsize=fontsize)

    if experts == ['DAPI', 'BIII', 'DAPI_BIII']:
        for j, expert in enumerate(experts):
            data_expert[expert] = auc_results[expert].dropna(axis=1).to_numpy().flatten()
            if scatter:
                color = 'black'
            else:
                color = marker_colors[expert]
            ax.boxplot(data_expert[expert], positions=[j], medianprops=dict(color=color, linewidth=linewidth * 2),
                       boxprops=dict(color=color, linewidth=linewidth), capprops=dict(color=color, linewidth=linewidth),
                       whiskerprops=dict(color=color, linewidth=linewidth), showfliers=False)
            for k, rbp in enumerate(RBPS[:-1]):
                data = auc_results[f'{rbp}_{expert}'].to_numpy()[0]
                if scatter:
                    ax.scatter(np.repeat(j, len(data)), data, marker=marker, c=marker_colors[rbp], edgecolors='none',
                               alpha=0.7,
                               label=f'subset {k}' if j == 0 else '_nolegend_')
            if j > 0:
                p = mannwhitney_ttest(('DAPI', data_expert['DAPI']), {f'{expert}': data_expert[f'{expert}']},
                                      alternative='less')
                if expert == 'BIII':
                    ax.text(j - 1, positions_p_value[0], f'p={p:.1e}', fontsize=fontsize)
                else:
                    ax.text(0.5, positions_p_value[1], f'p={p:.1e}', fontsize=fontsize)
                if expert == 'DAPI_BIII':
                    p = mannwhitney_ttest(('BIII', data_expert['BIII']), {f'{expert}': data_expert[f'{expert}']},
                                          alternative='less')
                    ax.text(j - 1, positions_p_value[2], f'p={p:.1e}', fontsize=fontsize)

        ax.set_xticklabels(experts, fontsize=fontsize)
        if scatter:
            ax.legend(labelspacing=0.1, handletextpad=0.2, borderpad=0.2, loc='lower right', fontsize=fontsize)

    else:
        if sorted:
            medians = []
            if 'DAPI_BIII_RBP' in experts:
                expert = 'DAPI_BIII_'
            elif experts == ['DAPI', 'RBP']:
                expert = ''
            for k, rbp in enumerate(RBPS[:-1]):
                data = auc_results[f'{expert}{rbp}'].to_numpy()[0]
                medians.append(np.median(data))
            sorted_indices = np.argsort(medians)
            list_rbp = [list(RBPS[:-1])[i] for i in sorted_indices]
        else:
            list_rbp = RBPS[:-1]

        if experts == ['DAPI_BIII', 'DAPI_BIII_RBP']:
            labels = []
            p_values = dict()
            for k, rbp in enumerate(list_rbp):
                data_expert = auc_results[f'{rbp}_DAPI_BIII'].to_numpy()[0]
                ax.boxplot(data_expert, positions=[3 * k], medianprops=dict(color='black', linewidth=linewidth * 2),
                           boxprops=dict(color='black', linewidth=linewidth),
                           capprops=dict(color='black', linewidth=linewidth),
                           whiskerprops=dict(color='black', linewidth=linewidth), showfliers=False)
                data = auc_results[f'DAPI_BIII_{rbp}'].to_numpy()[0]
                ax.scatter(np.repeat(3 * k + 1, len(data)), data, marker=marker, c=marker_colors[rbp],
                           edgecolors='none', alpha=0.7, label=rbp)
                ax.boxplot(data, positions=[3 * k + 1], medianprops=dict(color='black', linewidth=linewidth * 2),
                           boxprops=dict(color='black', linewidth=linewidth),
                           capprops=dict(color='black', linewidth=linewidth),
                           whiskerprops=dict(color='black', linewidth=linewidth), showfliers=False)

                p = mannwhitney_ttest(('DAPI_BIII', data_expert), {f'DAPI_BIII_{rbp}': data}, alternative='less')
                p_values[f'DAPI_BIII_{rbp}'] = p
                ax.text(3 * k, 1, f'p={p:.1e}', fontsize=fontsize)
                labels.append('DAPI\nBIII')
                labels.append(f'DAPI\nBIII\n{rbp}')
            ax.set_xticks([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])
            ax.set_xticklabels(labels, fontsize=fontsize)

        elif experts == ['DAPI', 'RBP']:
            p_values = dict()
            data_expert = auc_results['DAPI'].dropna(axis=1).to_numpy().flatten()
            ax.boxplot(data_expert, positions=[0], medianprops=dict(color='black', linewidth=linewidth * 2),
                       boxprops=dict(color='black', linewidth=linewidth),
                       capprops=dict(color='black', linewidth=linewidth),
                       whiskerprops=dict(color='black', linewidth=linewidth), showfliers=False)
            for k, rbp in enumerate(list_rbp):
                data_dapi = auc_results[f'{rbp}_DAPI'].to_numpy()[0]
                ax.scatter(np.repeat(0, len(data_dapi)), data_dapi, marker=marker, c=marker_colors[rbp],
                           edgecolors='none', alpha=0.7)
                data = auc_results[f'{rbp}'].to_numpy()[0]
                ax.scatter(np.repeat(k + 1, len(data)), data, marker=marker, c=marker_colors[rbp], edgecolors='none',
                           alpha=0.7, label=rbp)
                ax.boxplot(data, positions=[k + 1], medianprops=dict(color='black', linewidth=linewidth * 2),
                           boxprops=dict(color='black', linewidth=linewidth),
                           capprops=dict(color='black', linewidth=linewidth),
                           whiskerprops=dict(color='black', linewidth=linewidth), showfliers=False)

                p = mannwhitney_ttest(('DAPI', data_dapi), {f'{rbp}': data}, alternative='less')
                p_values[rbp] = p
                ax.text(k, 1, f'p={p:.1e}', fontsize=fontsize)
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_xticklabels(['DAPI'] + list_rbp, fontsize=fontsize)

        ax.legend(labelspacing=0.1, handletextpad=0.2, borderpad=0.2, loc='lower right', fontsize=fontsize)

    print()
    if 'stress' in classification and experts == ['DAPI_BIII', 'DAPI_BIII_RBP']:
        plt.ylim([0.8, 1.02])
    else:
        plt.ylim([0.48, 1.06])
    plt.tight_layout()
    if plot_pvalue:
        fig_p_value = log10_plot(p_values, title)
        plt.show()
        return fig, fig_p_value
    else:
        plt.show()
        return fig


def get_probabilities(expert: str, label: str, condition: str) -> pd.DataFrame:
    """Read probabilities from csv result file"""
    image_probabilities = pd.read_csv(f'{os.path.join(utils.get_project_root(), "results")}/image_probabilities.csv')
    probabilities = image_probabilities[f'expert_{expert}'].dropna()
    probabilities = probabilities[image_probabilities['als_label'] == label]
    probabilities = probabilities[image_probabilities['condition'] == condition]
    return probabilities


def plot_probabilities_per_expert(classifier: str, label: str, conditions: List[str],
                                  experts: List[str], figsize: (int, int)) -> plt.figure:
    """Violin plots of probabilities with one subplot per expert"""

    figsize = (cm_to_inch(figsize[0]), cm_to_inch(figsize[1]))
    fontsize = 8
    linewidth = 0.5

    fig, axs = plt.subplots(1, len(experts), figsize=figsize, sharey=True)
    if classifier == 'control_als_untreated':
        axs[0].set_ylabel('Probability of ALS', fontsize=fontsize)
    else:
        axs[0].set_ylabel('Probability of stress', fontsize=fontsize)

    for i, expert in enumerate(experts):
        print(expert)
        title = expert.replace('_', ' ')
        axs[i].axhline(y=0.5, alpha=0.5, linestyle='--', lw=1, color='black')
        labels = [f'{acronyms[condition]}\nCTRL' for condition in conditions]
        labels.insert(1, 'UT\nALS')

        p_values = dict()
        for j, condition in enumerate(conditions):
            color = condition_colors[condition]
            data = get_probabilities(f'{classifier}_{expert}', label, condition)
            print(condition, label, len(data), 'images')
            position = j

            if condition == 'untreated':
                data_als = get_probabilities(f'{classifier}_{expert}', 'als', 'untreated')
                print(condition, 'als', len(data_als), 'images')
                violin = axs[i].violinplot(data_als, positions=[1], showmeans=True)
                violin['bodies'][0].set_facecolor(color)
                violin['cmeans'].set_edgecolor('#000000')
                violin['cmeans'].set_linewidth(linewidth * 2)
                for param in ['cmins', 'cmaxes', 'cbars']:
                    violin[param].set_edgecolor(color)
                    violin[param].set_linewidth(linewidth)
            else:
                position = j + 1

            violin = axs[i].violinplot(data, positions=[position], showmeans=True)
            violin['bodies'][0].set_facecolor(color)
            violin['cmeans'].set_edgecolor('#000000')
            violin['cmeans'].set_linewidth(linewidth * 2)
            for param in ['cmins', 'cmaxes', 'cbars']:
                violin[param].set_edgecolor(color)
                violin[param].set_linewidth(linewidth)

            p = kolmogorovsmirnov_ttest(('untreated', data_als), {f'{condition}': data})
            p_values[condition] = p
            # axs[i].text(position, 1, f'p={p:.1e}', fontsize=fontsize)

        axs[i].set_title(title, fontsize=fontsize)
        axs[i].set_xticks(range(len(labels)))
        axs[i].set_xticklabels(labels, fontsize=fontsize)
        axs[i].yaxis.set_tick_params(labelsize=fontsize)
        print()

    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.show()
    return fig


def plot_probabilities_per_condition(classifier: str, label: str, conditions: List[str],
                                     experts: List[str], figsize: (int, int)) -> plt.figure:
    """Violin plots of probabilities with one subplot per condition"""

    figsize = (cm_to_inch(figsize[0]), cm_to_inch(figsize[1]))
    fontsize = 8
    linewidth = 0.5

    fig, axs = plt.subplots(1, len(conditions), figsize=figsize, sharey=True)
    for i, condition in enumerate(conditions):
        print(condition)
        title = f'{condition} stress'
        axs[i].set_title(title)
        axs[i].axhline(y=0.5, alpha=0.5, linestyle='--', lw=1, color='black')
        axs[0].set_ylabel('Probability of ALS for CTRL images', fontsize=fontsize)

        labels = []
        means = []
        axs[i].axhline(y=0.5, alpha=0.5, linestyle='--', lw=1, color='black')
        for j, expert in enumerate(experts):
            data = get_probabilities(f'{classifier}_{expert}', label, condition)
            means.append(data.mean())

        sorted_indices = np.argsort(means)
        list_rbp = [RBPS[i] for i in sorted_indices]

        data = get_probabilities(f'{classifier}_DAPI_BIII', label, condition)
        print('DAPI_BIII', label, len(data), 'images')
        color = marker_colors['DAPI_BIII']
        violin = axs[i].violinplot(data, positions=[0], showmeans=True)
        violin['bodies'][0].set_facecolor(color)
        violin['cmeans'].set_edgecolor('#000000')
        violin['cmeans'].set_linewidth(linewidth * 2)
        for param in ['cmins', 'cmaxes', 'cbars']:
            violin[param].set_edgecolor(color)
            violin[param].set_linewidth(linewidth)
        labels.append(f'DAPI\nBIII')

        for j, rbp in enumerate(list_rbp):
            color = marker_colors[rbp]
            expert = f'{classifier}_DAPI_BIII_{rbp}'
            data = get_probabilities(expert, label, condition)
            print(f'DAPI_BIII_{rbp}', label, len(data), 'images')
            violin = axs[i].violinplot(data, positions=[j + 1], showmeans=True)
            violin['bodies'][0].set_facecolor(color)
            violin['cmeans'].set_edgecolor('#000000')
            violin['cmeans'].set_linewidth(linewidth * 2)
            for param in ['cmins', 'cmaxes', 'cbars']:
                violin[param].set_edgecolor(color)
                violin[param].set_linewidth(linewidth)
            labels.append(f'DAPI\nBIII\n{rbp}')

        axs[i].set_title(title, fontsize=fontsize)
        axs[i].set_xticks(range(len(labels)))
        axs[i].set_xticklabels(labels, fontsize=fontsize)
        axs[i].yaxis.set_tick_params(labelsize=fontsize)
        print()
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.show()
    return fig


def log10_plot(p_values: Dict, title: str) -> plt.figure:
    """Bar plot log10 of p-values"""
    fig, ax = plt.subplots(figsize=(cm_to_inch(7), cm_to_inch(5)))
    sorted_experts = sorted(p_values, key=p_values.get, reverse=True)
    labels = []
    for i, expert in enumerate(sorted_experts):
        if expert not in RBPS[:-1]:
            rbp = expert[10:]
        else:
            rbp = expert
        log10_p = -math.log10(p_values[expert])
        ax.bar(x=i, height=log10_p, color=marker_colors[rbp], alpha=0.5)
        labels.append(expert.replace('_', '\n'))
    plt.axhline(y=-np.log10(0.05), linestyle='--', color='grey', label=f'\u03B1=0.05')
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('-log10(p-value)', fontsize=8)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylim([0, 4.2])
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend()
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    return fig


def plot_probabilities_stress_recovery(stress: str, label: str, experts: List[str], figsize: (int, int)):
    """Violin plots of probabilities for stress recovery with one subplot per expert"""
    classifier = f'control_untreated_{stress}'
    figsize = (cm_to_inch(figsize[0]), cm_to_inch(figsize[1]))
    fontsize = 8
    linewidth = 0.5

    conditions = {'osmotic': ['untreated', 'osmotic', 'osmotic_1h', 'osmotic_2h', 'osmotic_6h'],
                  'heat': ['untreated', 'heat', 'heat_2h']}

    fig, axs = plt.subplots(1, 5, figsize=figsize, sharey=True)
    title = f'{stress} stress'
    plt.suptitle(title, fontsize=fontsize)
    axs[0].set_ylabel(f'Probability of stress for {acronyms[label]} images', fontsize=fontsize)

    for i, rbp in enumerate(RBPS[:-1]):
        axs[i].axhline(y=0.5, alpha=0.5, linestyle='--', lw=1, color='black')
        axs[i].set_ylim([-0.05, 1.05])
        if experts == 'DAPI_BIII_RBP':
            expert = f'DAPI_BIII_{rbp}'
        elif experts == 'RBP':
            expert = rbp

        print(expert)
        for j, condition in enumerate(conditions[stress]):
            data = get_probabilities(f'{classifier}_{expert}', label, condition)
            print(condition, len(data))
            violin = axs[i].violinplot(data, positions=[j], showmeans=True)
            violin['bodies'][0].set_facecolor(marker_colors[rbp])
            violin['cmeans'].set_edgecolor('#000000')
            violin['cmeans'].set_linewidth(linewidth * 2)
            for param in ['cmins', 'cmaxes', 'cbars']:
                violin[param].set_edgecolor(marker_colors[rbp])
                violin[param].set_linewidth(linewidth)
        print()
        labels = [acronyms[condition] for condition in conditions[stress]]
        axs[i].set_xticks(range(len(labels)))
        axs[i].set_xticklabels(labels, fontsize=fontsize)
        axs[i].set_title(expert.replace('_', ' '), fontsize=fontsize)
        axs[i].yaxis.set_tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.show()
    return fig
