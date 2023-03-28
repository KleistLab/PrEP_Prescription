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

## 2 Execute Code
All scripts that were used to create the results from the provided datasets can be found under 
`<path_to_PrEP_Prescription>/factories/`. Since all intermediate results are provided by the repository, each script can 
be executed seperately.
* `00_preliminary_data_creation`: Transforms the provided prescription data, as described in section [...], and fits the 
model against it. The script creates the files `/data/prescriptions_daily.tsv` and `/data/model_parameters.tsv`.
* `01_data_sampling`: Samples datasets from a binomial distribution. Number of samples per state can be set with variable `n_samples`. 
The resulting dataset will be stored under `/results/data_sampling/<n_samples>/<current_date>/prescriptions_daily_sampled.bin`.
To use the generated dataset for subsequent steps, i.e. model fitting, copy the generated file into the data folder, or set `save_in_data_folder = True` before executing the code. 
This will store the generated file automatically in the data folder.
* `02_model_fitting`: Fits the model against each data sample and creates `model_parameters_sampled.tsv`, containing model parameters for each parameter.
The model parameters will be stored under `/results/model_parameters_sampled.tsv`. To use these parameters for the final analysis, copy the file into the data folder, or set `save_in_data_folder = True` before executing the code.
This will store the generated file automatically in the data folder.
* `03_evaluation`: Creates all figures and different model predictions, such as estimated number of PrEP users at different times. 
All figures will be stored under `/results/figures/` and all model predictions are stored in `/results/milestones.tsv`


## 2. Workflow