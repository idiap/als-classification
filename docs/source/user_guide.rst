==========
User Guide
==========

This user guide is also available in the form of a jupyter notebook in ``notebooks`` > ``user_guide.ipynb``

Main concepts
=============

There are 3 classes in this package:

* Class ``Database``: defines a pandas dataframe containing annotations to select specific images according to a protocol

* Class ``Protocol``: defines the rules to select images in the database.

* Class ``CustomDataset``: defines a custom dataset in pytorch to train the model on this dataset. The dataset is defined according to a protocol.

With these 3 classes and the functions in ``src.utils``, you can train and/or evaluate a model


Database
========

A ``Database`` object has several attributes:

* ``annotations``: annotations for the fluoMNs dataset, with all available images and their specifications. One row corresponds to one field of view of a well, with several channels and several planes. The ``exclude`` column indicates whether the sample should be excluded from the experiments, due to bad experimental conditions for example.

* ``all_samples``: all *valid* samples (annotations with ``exclude == no``)

* ``protocols``: all available protocols defined for this database


Protocol
========

A ``Protocol`` defines which data are selected in the database. When initializing a protocol, we specify which **labels**, which **conditions** and which **RBP** we want.

The **label** tells whether the cell line in the cell culture is healthy (``'control'``) or als-mutant (``'als'``). The **condition** corresponds to the stress condition under which the cell culture is put (e.g. ``'untreated'`` if the cell is not put under stress). The **RBP** (stands for RNA-binding protein) is the fluorescent marker that we choose. ``'all'`` stands for all RBPs (used when we want the DAPI or BIII markers because all RBP subsets have the DAPI and BIII channels).


Here are the different possibilities:

.. code-block:: sh

    labels:  ['control', 'als']
    conditions:  ['untreated', 'oxidative', 'heat', 'heat_2h', 'osmotic', 'osmotic_1h', 'osmotic_2h', 'osmotic_6h']
    rbps:  ['SFPQ', 'TDP-43', 'FUS', 'hnRNPA1', 'hnRNPK', 'all']

A protocol can have either one or two labels, and one or two conditions.

To classify between healthy and als-mutant cells, you need both ``'als'`` and ``'control'`` in your protocol. To classify between unstressed and stressed cells, you need both ``'untreated'`` and a stressor such as ``'osmotic'`` in your protocol.

Here is an example ``protocol``, with images of control and als untreated motor neurons, where the ``'TDP-43'`` channel must be available. **Note**: this protocol corresponds to the images available in the data subset which can be downloaded on Zenodo **[link]**.

.. code-block:: sh

    protocol = Protocol(['control', 'als'], ['untreated'], 'TDP-43')

Each ``protocol`` has a ``name`` method, which corresponds to ``labels_conditions_rbp``. Please not that labels and conditions are always sorted in reverse alphabetical order.

.. code-block:: sh

    'control_als_untreated_TDP-43'



CustomDataset
=============

When using Pytorch for training and testing, you need a dataset class over which Pytorch will iterate to find your images. See this `tutorial <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_ for an example.  Our ``CustomDataset`` represents the actual dataset of images that we will use either for training or testing.

To keep it short, this class selects some indices from a protocol (could be either indices from the train set or from the test set), finds the corresponding images in the source directory, applies some preprocessing on those images and stores them in a target directory.

**CAUTION** : Every time you create a ``CustomDataset``, a target directory is created and starts storing images in it (if you access elements in the dataset). This can take quite some space so don't forget to use the ``delete`` method to delete the target directory when you are done with your experiment.

To actually use this dataset in training and testing, we use Pytorch dataloaders (defined in ``src.utils``) to iterate over the elements.


Training
========

There are 52 trainable models in total, 13 for each of the following binary classifications:

* ``'als'`` vs ``'control'``,
* ``'untreated'`` vs ``'osmotic'`` stress,
* ``'untreated'`` vs ``'oxidative'`` stress, and
* ``'untreated'`` vs ``'heat'`` stress

The 13 models correspond to different channels combinations that we want to compare:

* ``'DAPI'``,
* ``'BIII'``,
* ``'DAPI-BIII'``,
* ``'SFPQ'``,
* ``'FUS'``,
* ``'TDP-43'``,
* ``'hnRNPA1'``,
* ``'hnRNPK'``,
* ``'DAPI-BIII-SFPQ'``,
* ``'DAPI-BIII-FUS'``,
* ``'DAPI-BIII-TDP-43'``,
* ``'DAPI-BIII-hnRNPA1'``,
* ``'DAPI-BIII-hnRNPK'``.

If you have GPU resources (strongly recommended), you can retrain the models using ``train_model.py``. It is a command-line application that you can call from the terminal with the desired options:

.. command-output:: python ../../src/train_model.py --help

For example, if you have downloaded the subset of data on Zenodo **[link]**, you can run the following line. It will train the model to classify images of ``'als'`` and ``'control'`` cultures of untreated motor neurons,  only using the ``'TDP-43'`` channel. With the available images from this subset, you can also use either ``'DAPI'``, ``'BIII'``, ``'DAPI-BIII'`` or ``'DAPI-BIII-TDP-43'``.

.. code-block:: sh

    python src/train_model.py -c user -cl als -p control_als_untreated_TDP-43 -ch TDP-43 -f 0 -s False

Training models will save results in ``results`` > ``auc.csv``. This file contains the AUC (measure of performance) on the test set for each fold in 10-fold cross validation for each protocol associated with channels.

Prediction
==========

If you don't have GPU resources or you simply want to evaluate already trained models on some data, you can use ``predict_model.py``. It is a command-line application that you can call from the terminal with the desired options:

.. command-output:: python ../../src/predict_model.py --help

For example, if you have downloaded the subset of data on Zenodo **[link]**, and the trained model entitled ``'state_dict_control_als_untreated_TDP-43_TDP-43_fold_0.pt'``, you can run the following line. It will evaluate the model which was trained to classify images of ``'als'`` and ``'control'`` cultures of untreated motor neurons using the ``'TDP-43'`` channel on images of ``'control'`` cultures of untreated motor neurons, which were not seen during training. You can also evaluate on images of ``'als'`` cultures of untreated motor neurons.

.. code-block:: sh

    python src/predict_model.py -c user -cl als -e TDP-43 -la control -co untreated

Evaluating models will save results in ``results`` > ``image_probabilities.csv``. This file contains the output probabilities from each of the 52 models, on each *valid* image of the dataset (i.e. models trained on images with the SFPQ channel are only evaluated on images containing this channel).

Figures
=======

Results from our experiments are stored in ``results`` > ``auc.csv``, ``image_probabilities.csv``. If you want to reproduce some figures, you can use functions in ``visualization`` > ``visualize_utils.py`` or simply generate figures from the terminal with ``visualize.py``: :

.. command-output:: python ../../src/visualization/visualize.py --help