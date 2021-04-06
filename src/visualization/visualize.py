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

import click
from src.visualization import visualize_utils


@click.command()
@click.option('-f', '--figure', type=click.Choice(['1D', '2A', '2D', '3A', '3C', '4A', '2C', '3D', '4B', '5A', '5B', '6A']),
                                required=True, help='''Choose the figure to generate''')
def generate_figure(figure):

    plot_function = getattr(visualize_utils, f'figure_{figure}')
    plot_function()


if __name__ == "__main__":
    generate_figure()
