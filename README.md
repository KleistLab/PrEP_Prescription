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

The sampled datasets (`prescriptions_daily_sampled.bin`) is too large to upload to github. However, it can be downloaded 
from [Nextcloud](https://fs.bmi.app.bmi.dphoenixsuite.de/s/tNLy567ERptYjR3?dir=undefined&openfile=29953) or created by 
executing `/factories/01_data_sampling.py`.

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

### Download file
Download `prescriptions_daily_sampled.bin` from [Nextcloud](https://fs.bmi.app.bmi.dphoenixsuite.de/s/tNLy567ERptYjR3?dir=undefined&openfile=29953) 
and copy into the data folder.

## 2 Reproduce Results
Under `<path_to_PrEP_Prescriptions>/factories/` there are four scripts that can be used to reproduce the results. As all
intermediate results are stored in the data folder, each script can be run individually. The scripts cover the four main
steps: (i) preliminary steps, such as initial data transformation and model fitting, (ii) data sampling, (iii) model 
fitting to the sampled data, and (iv) final evaluation of the results, including plotting and calculation of some statistics.

***Important Note:***\
*Since a large portion of prescribed TDF/FTC are 90-pill packages, each datapoint in the continuous dataset depends on the
previous 90 days. As a consequence, the first 90 timepoints in a continuous dataset lack information and should be removed
when fitting a model against it.\
Data used in this project covers a period from January 2017 until December 2021. However, because of what we have described above,
the final models and their parameters assume a start date of April 1st 2017., i.e. $t_0 = Jan.\ 1st\ 2017$.*


### Preliminary Steps
Script:
```/factories/01_preliminary_data_creation.py```

In this step, continuous trajectories are created from the provided prescription data and the model is fitted onto them.
This is necessary because the data sampling step relies on the estimated number of TDF/FTC prescriptions for HIV therapy.\
The transformed prescription data are stored under `/data/prescriptions_daily.tsv` and the parameters of the fitted model 
under `/data/model_parameters.tsv`.

### Data Sampling
Script: ```/factories/02_data_sampling.py```

This script samples datasets from a binomial distribution using information from the prescription dataset and the model 
created in the previous step. The distributions from which the datasets are sampled are described in the Methods section.
Datasets are then stored as an `xarray.Dataarray` object with dimensions 'state', 'sample', 'year', 'month' and 'day'.\
*Note: Since numpy arrays require each inner array to have the same shape, the arrays corresponding to the 'day' dimension
are always of size 31. Entries corresponding to non-existing dates (e.g. Feb. 31st) are encoded as `NaN`.*\

The resulting `Dataarray` object is stored under `/results/data_sampling/<n_samples>/<current_date>/prescriptions_daily_sampled.bin`,
using `pickle`.
To use the generated dataset for subsequent steps, i.e. model fitting, the generated file must be copied into the data
folder. Alternatively, `save_in_data_folder = True` can be set before running the code. This will save the generated 
file automatically into the data folder, but also overwrites the previous file.

### Model Fitting
Script: ```/factories/03_model_fitting.py```

In this step, the model is fitted against each sampled dataset, resulting in a parameter set for each sample. Since the
success and runtime of model optimization highly depends on the initial parameter guesses, the model fitting of each 
federal state consists of two steps:
1. The median of all samples from a federal state is computed. The model is then fitted against this median dataset, using
100 sets of initial parameter guesses, sampled from a log-uniform distribution.\
(Results of this first optimization step are stored under `/results/optimization_results/<current_date>/optimization_<federal state>.tsv`) 
2. Using the best optimal parameter set from the previous step as initial guess, the model is then fitted against each data sample. 

This not only ensures success for most datasets, but also speeds up the model optimization process.

The model parameters will be stored under `/results/model_parameters_sampled.tsv`. To use these parameters for the final
analysis, copy the file into the data folder. Alternatively, `save_in_data_folder = True` can be set before running the 
code. This will save the generated file automatically into the data folder, but also overwrites the previous file.


### Evaluation of Results
Script: `/factories/04_evaluation.py`

Finally, the generated models are then evaluated. This script creates several figures, as well as some stats:
* Plots the sampled datasets
* Plots TDF/FTC prescription numbers for PrEP and ART, as predicted by the models
* Computes PrEP users from prescription numbers and plots them
* Computes absolute and relative prescription numbers at different times

All figures are stored under `/results/figures/` and all model predictions are stored in `/results/milestones.tsv`

## 3 Methods

### Generation of continuous trajectories from monthly prescription data
Our data set contained the number of Truvada prescriptions per month for the different package sizes available in Germany. 
For each prescription we drew a random date within the month it was prescribed and incremented the next k days by one, where 
k denotes the prescribed package size. Using this procedure we obtain a trajectory of daily Truvada coverage, assuming that 
Truvada was taken daily for treatment or PrEP.

### Mathematical model
The model used in this project consists of two equations Y<sub>ART</sub> and Y<sub>PrEP</sub> that model the prescription 
numbers for HIV therapy and PrEP, respectively.
```math 
\begin{align*}
    \frac{dY_{ART}(t)}{dt} &= k_{ART}\cdot Y_{ART}(t)\\
    \frac{dY_{PrEP}(t)}{dt} &= k_{PrEP}(t)(N_{inNeed} - (c_{on-demand}\cdot c_{SHI}\cdot Y_{PrEP}(t))\\
    \frac{dY_{tot}(t)}{dt} &= Y_{ART}(t) + Y_{PrEP}(t)\\
\end{align*}
```
with initial values $Y_{ART}(t_0) = Y_{ART,0}$  and $Y_{PrEP}(t_0) = Y_{PrEP,0}$. For $Y_{ART}$, we assume an exponential decay, 
reflecting the slow decline of TDF/FTC use in HIV therapy. In the case of PrEP prescriptions, $Y_{PrEP}$, we assume that
they tend to increase over time and may eventually saturate when the number of people in need of PrEP ($N_{inNeed}$) is 
reached.

We assumed that PrEP uptake, reflected by parameter k_{PrEP}(t), changes between distinct episodes: 

1. Jan 1st 2017 – Aug 31st 2019  (before coverage by insurance)
2. Sep. 1st 2019  - Nov 30th 2019 (initial run on PrEP)
3. Dec. 1st 2019 – Mar 31st 2020 (before 1st Lockdown)
4. Apr. 1st 2020 – Jun 30th 2020 (right after / during 1st Lockdown)
5. Jul. 1st 2020 – Nov. 30th 2020 (before 2nd lockdown)
6. Dec. 1st 2020 – Feb 28th 2021 (right after / during 2nd lockdown)
7. Mar. 1st 2021 -

### Model Fitting
To obtain the model parameters and initial values, the model is fitted to the number of TDF/FTC prescriptions, normalized 
by package size, by minimizing the residual sum of squares (RSS):   
```math
\begin{align*}
    &\min{x} || y(t) – f(t, x) ||_2^2\\
    &\text{where } f(t, x) = Y_{tot}(t) = Y_{ART}(t, k_{ART}, Y_{ART,0}) + Y_{PrEP}(t, k_{PrEP}(t), Y_{PrEP,0})
\end{align*}
```
Parameters are determined for the individual federal states, as well as for the entire country.

### Data Sampling

To estimate uncertainty in the data, parameters and model predictions, we perform a parametric re-sampling technique. 
This is done in a two-step process.

First, the total number of TDF/FTC prescriptions per month (N_hatTDF/FTC(t)) is sampled from a binomial distribution:
```math
\hat{N}_{TDF/FTC}(t) \sim B(N_{TDF/FTC},\ p_{TDF/FTC})
```
,where the number of TDF/FTC prescriptions $N_{TDF/FTC}(t) = N_{inNeed} + Y_{ART}(t)$. $Y_{ART}(t)$ was estimated by the
model fitted against the prescription data. $p_{TDF/FTC} = \frac{N_{30}(t) + N_{90}(t)}{N_{TDF/FTC}(t)}$,the probability 
of a TDF/FTC prescription at time $t$, is estimated from the number of prescribed 30 and 90 pill packages at time t, provided 
by the dataset.

In a second step the number of 30- and 90-pill prescriptions are sampled:
```math
\begin{align*}
&\hat{N}_{30}(t) \sim B(N_{TDF/FTC}(t),\ p_{30}(t)) \\
&\hat{N}_{90}(t) = \hat{N}_{TDF/FTC}(t) - \hat{N}_{30}(t)
\end{align*}
```
where $p_{30} = \frac{N_{30}(t)}{N_{30}(t) + N_{90}(t)}$ is the probability for a 30-pill prescription at time t.

### Translating number of prescription to PrEP users
To estimate the actual number of PrEP users from the estimated number of prescriptions, we take intermitted/on-demand use 
into account. In addition, not everyone is covered by the NHS, and some PrEP users are privately insured or self-payers.
In a previous study, Schmidt et al. reported 18.9% on-demand users. The average number of prescribed PrEP pills divided
by the number of days of PrEP use was reported to be 0.58 for on-demand users and 0.91 for daily users [Schmidt et al., Infection 2022]. 
The average number of PrEP users covered by SHI was reported to be 89.5% [Schmidt et al., Epidemiologisches Bulletin 2021].

This results to the following calculation of PrEP users:
```math
N_{PrEPUsers}(t) = (\frac{0.189}{0.58} + \frac{0.811}{0.91}) * \frac{1}{0.895} * Y_{PrEP}(t) = c_{on-demand} * c_{SHI} * Y_{PrEP}(t)
```
