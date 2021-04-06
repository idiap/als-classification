
==========
Python API
==========

Protocol
--------

.. autosummary::
    :toctree: api/protocol

    src.protocol.Protocol

Database
--------

.. autosummary::
    :toctree: api/database

    src.database.Database

Dataset
--------

.. autosummary::
    :toctree: api/dataset

    src.dataset.CustomDataset

Train
-----

.. click:: src.train_model:classify
    :prog: classify
    :nested: full


Test
----

.. click:: src.predict_model:evaluate
    :prog: evaluate


Visualization
-------------

.. click:: src.visualization.visualize:generate_figure
    :prog: generate_figure











