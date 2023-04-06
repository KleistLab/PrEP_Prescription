# PrEP_Prescription
Repository for the Truvada prescription data analysis in Germany, as well as PrEP usage.

The repository contains all code to reproduce the results from [insert publication/preprint]. It consists of four steps,
which are described in detail in section [...]:
1. Preliminary Data Creation
2. Data Sampling
3. Model Fitting
4. Evaluation

Further, the prescription data, as well as all intermediate results (except for the sampled data) are stored in the data folder.
The following files can be found there:
* `population_at_risk.tsv`: Contains the estimated number of people in need of PrEP (col: Total)
* `prescription_data.tsv`: The initial prescription data file
* `prescriptions_daily.tsv`: Transformed version of prescription data, i.e. continuous trajectories
* `model_parameters.tsv`: Parameters of the model fitted against the data in `prescriptions_daily.tsv`, starting Jan. 1st 2017.
* `model_parameters_sampled.tsv`: Parameters of the models fitted against the sampled data, starting Apr. 1st 2017.

The sampled datasets (`prescriptions_daily_sampled.bin`) is too large to upload to github. However, it can be created by 
executing `/factories/01_data_sampling.py`. It is also possible to download it from [Nextcloud](https://fs.bmi.app.bmi.dphoenixsuite.de/s/tNLy567ERptYjR3?dir=undefined&openfile=29953).

## 1 Installation
### Clone Repository
```
git clone https://github.com/KleistLab/PrEP_Prescription
```

### Create Virtual Environment
```
mkvirtualenv prep_prescription --python=python3.9
workon prep_prescription
```

### Install
Install `PrEP_Prescription` in the newly created virtual environment.
```
(prep_prescrption) cd PrEP_Prescription
(prep_prescrption) pip install -e . 
```

## 2 Reproduce Results
Under `<path_to_PrEP_Prescriptions>/factories/` there are four scripts that can be used to reproduce the results. As all
intermediate results are stored in the data folder, each script can be run individually. The scripts cover the four main
steps: (i) preliminary steps, such as initial data transformation and model fitting, (ii) data sampling, (iii) model 
fitting to the sampled data, and (iv) final evaluation of the results, including plotting and calculation of some statistics.

### Preliminary Steps
Script:
```/factories/00_preliminary_data_creation.py```\

In this step, continuous trajectories are created from the provided prescription data and the model is fitted onto them.
This is necessary because the data sampling step relies on the number of TDF/FTC prescriptions for HIV therapy.\
The transformed prescription data are stored under `/data/prescriptions_daily.tsv` and the parameters of the fitted model 
under `/data/model_parameters.tsv`.

### Data Sampling
Script: ```/factories/01_data_sampling.py```\

This script samples datasets from a binomial distribution using information from the prescription dataset and the model 
created in the previous step. The distributions from which the datasets are sampled are described in the Methods section.
Datasets are then stored as an `xarray.Dataarray` object with dimensions 'state', 'sample', 'year', 'month', 'day'.\
*Note: Since numpy arrays require each inner array to have the same shape, the arrays corresponding to the 'day' dimension
are always of size 31. Entries corresponding to non-existing dates (e.g. Feb. 31st) are encoded as `NaN`.*\

The resulting `Dataarray` object is stored under `/results/data_sampling/<n_samples>/<current_date>/prescriptions_daily_sampled.bin`,
using `pickle`.
To use the generated dataset for subsequent steps, i.e. model fitting, the generated file must be copied into the data
folder. Alternatively, `save_in_data_folder = True` can be set before running the code. This will save the generated 
file automatically into the data folder, but also overwrites the previous file.

### Model Fitting
Script: ```02_model_fitting.py```\
In this step, the model is fitted against each sampled dataset, resulting in a parameter set for each sample. Since the
success and runtime of model optimization highly depends on the initial parameter guesses, the model fitting of each 
federal state consists of two steps:
1. The median of all samples from a federal state is computed. The model is then fitted against this median dataset, using
100 sets of initial parameter guesses, sampled from a log-uniform distribution. 
2. Using the best optimal parameter set from the previous step as initial guess, the model is then fitted against each data sample. 

This not only ensures success for most datasets, but also speeds up the model optimization process.\

The model parameters will be stored under `/results/model_parameters_sampled.tsv`. To use these parameters for the final
analysis, copy the file into the data folder. Alternatively, `save_in_data_folder = True` can be set before running the 
code. This will save the generated file automatically into the data folder, but also overwrites the previous file.

*** TOADD: REMOVE THE FIRST 90 DAYS FROM DATASETS WHEN FITTING MODEL AGAINST IT ***
### Evaluation of Results
Finally, the generated models are then evaluated. This script creates several figures, as well as some stats:
* Plots the sampled datasets
* Plots TDF/FTC prescription numbers for PrEP and ART, as predicted by the models
* Computes PrEP users from prescription numbers and plots them
* Computes absolute and relative prescription numbers at different times

All figures are stored under `/results/figures/` and all model predictions are stored in `/results/milestones.tsv`


## 3 Methods

### Mathematical Model
The model used in this project consists of two equations Y<sub>ART</sub> and Y<sub>PrEP</sub> that model the prescription 
numbers for HIV therapy and PrEP, respectively.$Y_A$

