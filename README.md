Classification of ALS and Stress in Cultures of Motor Neurons
=============================================================

This project aims to increase our understanding of **Amyotrophic Lateral Sclerosis (ALS)**, which is a neurodegenerative disease. 
It deals with the classification of cultures of human induced Pluripotent Stem Cells (iPSCs) motor neurons. 
The neurons are either healthy, or contain the ALS mutation, and are put under different cellular stress (osmotic, oxidative and heat). 

The project is described in a paper submitted on BiorXiv which you can find **here** [TBA]


## Installation 

Here are the steps to follow to be able to use this package:

1. Install working environment
2. Download the database
3. Adapt configuration file 
4. Download trained models

#### Working environment 
Installation requirements can be found in `requirements.txt`.

Those can be installed within a `conda` environment (recommended):

```bash
conda create -n als # empty environment
conda activate als
conda env update -f environment.yml # add packages from requirements.txt
```


#### Database 

The database is separated from the package and can be downloaded **here** [TBA]. 
**CAUTION**: the dataset is very large (~1TB) so make sure you have enough space to store it. 
You can either put it in the `data` folder or choose your own location. In the next step, you will enter the location of the folder you chose in the configuration file. 

**Note**: The complete database is not available on IDR yet, but a subset of the database can be downloaded [**here**](https://zenodo.org/record/4664177) in the meantime. 


#### Configuration file

Open `scripts`>`config_user.yml`. In this file, enter a value for each field (batch_size, n_epochs, learning_rate) and in particular, enter the ABSOLUTE path of the folder where you downloaded the database for the `source_directory`. The `target_directory` is used during training and testing to create temporary directories of images. Again, this usually requires a large storage size (several GB) so make sure you have the available space in the directory you choose. 

When you use a command and need to choose a config, please choose `user`. You can configure other config files and use those as well (e.g. to train models separately on a grid). 

#### Models

Models can be quite long to train given the large number of images. If you want to evaluate models which are already trained, you can find them [**here**](https://www.idiap.ch/resource/fluoMNs_models). Put the downloaded models under `models`. 


## Getting Started

See `notebooks`>`user_guide.ipynb` 


## Questions

Refer to the documentation and/or source code. Go to `docs` > `source`. Run the following: 
```bash
make html
```
Go to `docs` > `build` > `html` and open `index.html` in your browser. 

## Tests

You can make sure everything is working correctly with the following command:
```bash
python -sv ../src/test/
```

## Contact 

For questions or reporting issues to this software package, contact the authors of the paper: colombine.verzat@idiap.ch or raphaelle.luisier@idiap.ch. 