# PrEP_Prescription
Repository for the Truvada prescription data analysis in Germany, as well as PrEP usage.

The repository contains all code to reproduce the results from [insert publication/preprint]. It consists of four steps,
which are described in detail in section [...]:
1. Preliminary Data Creation
2. Data Sampling
3. Model Fitting
4. Evaluation

## 1. Run Code
### 1.1 Install Dependencies
The following python packages need to be installed, using `pip install <package>`:
* numpy, pandas, scipy
* ray
* xarray

### 1.2 Execute Code
All scripts that were used to create the results from the provided datasets can be found under 
`<path_to_PrEP_Prescription>/factories/`. Since all intermediate results are provided by the repository, each script can 
be executed seperately.
* `00_preliminary_data_creation`: Transforms the provided prescription data, as described in section [...], and fits the 
model against it. It is possible to set the number of initial parameter guesses during model optimization, by changing the variable `n_p_init` in line 25.