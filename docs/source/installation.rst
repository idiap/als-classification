============
Installation
============

Here are the steps to follow to be able to use this package:

1. Install working environment
2. Download the database
3. Adapt configuration file
4. Download trained models

Working environment
^^^^^^^^^^^^^^^^^^^
Insallation requirements can be found in ``requirements.txt``.

Those can be installed within a ``conda`` environment (recommended):

.. code-block:: sh

    conda create -n als # empty environment
    conda activate als
    conda env update -f environment.yml # add packages from requirements.txt


Database
^^^^^^^^

The `complete database`_ is separated from the package and can be downloaded on `IDR`_.
**CAUTION**: the dataset is very large (~1TB) so make sure you have enough space to store it.
You can either put it in the ``data`` folder or choose your own location. In the next step, you will enter the location of the folder you chose in the configuration file.

**Note**: The complete database is not available on IDR yet, but a `subset`_ of the database can be downloaded on `Zenodo`_ in the meantime.

Configuration file
^^^^^^^^^^^^^^^^^^

Open ``scripts`` > ``config_user.yml``. In this file, enter a value for each field (batch_size, n_epochs, learning_rate) and in particular, enter the ABSOLUTE path of the folder where you downloaded the database for the ``source_directory``. The ``target_directory`` is used during training and testing to create temporary directories of images. Again, this usually requires a large storage size (several GB) so make sure you have the available space in the directory you choose.

When you use a command and need to choose a config, please choose ``user``. You can configure other config files and use those as well (e.g. to train models separately on a grid).

Models
^^^^^^

Models can be quite long to train given the large number of images. If you want to evaluate `models`_ which are already trained, you can find them on `Zenodo`_. Put the downloaded models under ``models``.


Tests
^^^^^

You can make sure everything is working correctly with the following command:

.. code-block:: sh

    python -sv ../src/test/


.. include:: links.rst