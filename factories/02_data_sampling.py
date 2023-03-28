"""
Data Sampling

The scripts computes the probabilities for 30- and 90-pill prescriptions for each timepoint t.
Using these probabilities, prescription numbers are then sampled from a binomial distribution.

The resulting dataset is stored as an xarray with dimensions state, sample, year, month, day and stored under
<path_to_PrEP_Prescription/results/data_sampling/<n_samples>/<date>/prescriptions_daily_sampled.bin as a binary file.

To use the generated dataset for model fitting, set the variable use_dataset = True, or copy the resulting binary file
manually into the data folder of the repository.

Variable n_samples controls the number of samples per state generated.
"""

import numpy as np
import pandas as pd

import datetime as dt
import pickle
import calendar

from src.sampling.sampling import sample_dataset
from src.models import prep_model
from src.optimization.simulate import Simulator
from src import DATA_PATH, RESULT_PATH


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


save_in_data_folder = False     # set to True to use the generated dataset for subsequent steps.
                                # NOTE: This will save the dataset under data/prescriptions_daily_sampled.bin and replace previously generated files!
n_samples = 100         # number of samples
start = (2017, 1, 1)    # earliest date to sample
n_nodes = 10            # number of nodes/threads to use

def main():

    df_raw = pd.read_csv(DATA_PATH / 'prescription_data.tsv', sep='\t')
    end_year, end_month = max(df_raw.apply(lambda x: dt.date(x['year'], x['month'], 1), axis=1)).timetuple()[:2]
    end_day = calendar.monthrange(end_year, end_month)[1]
    end_date = dt.date(end_year, end_month, end_day)
    end = (end_year, end_month, end_day)

    # ESTIMATE NUMBER OF TDF/FTC PRESCRIPTIONS FOR ART
    ## initialize Model and Simulator
    model = prep_model
    sim = Simulator(model)

    ## get model parameters
    df_parameters = pd.read_csv(DATA_PATH / 'model_parameters.tsv', sep='\t')
    df_parameters['sim_endpoints'] = df_parameters.sim_endpoints.apply(
        lambda x: [int(i) for i in x[1:-1].split(',')])  # convert string to list[int] object

    ## get indices of last days in month
    date_range = np.arange(dt.date(2017, 1, 1), end_date + dt.timedelta(days=1)).astype(dt.date)    # all dates from 2017-1-1 to 2021-12-31
    last_days = [dt.date(y, m, calendar.monthrange(y, m)[1]) for y in range(2017, end_year + 1) for m in range(1, 13)]  # last day of each month within that timerange
    last_days_idx = [np.where(date_range == d)[0][0] for d in last_days]
    year_arr = [int(date.year) for date in last_days]
    month_arr = [int(date.month) for date in last_days]

    ## simulate model for each state
    sim_dicti = {
        'state': np.array([]),
        'year': np.array([], dtype=int),
        'month': np.array([], dtype=int),
        'y_art': np.array([]),
    }
    for _, row in df_parameters.iterrows():
        state = row['state']
        endpoints = row['sim_endpoints']
        t_step = row['sim_t_step']
        y0 = []
        p = []
        for l in range(len(endpoints)):
            y0.append([row[f"{fid}{l}"] for fid in model.fids])
            p.append([row[f"{pid}{l}"] for pid in model.pids])
        sim_results = sim.simulate_continuous(endpoints, t_step, p, y0)
        y_art = sim_results.y[0][last_days_idx]     # select only numbers for last day of each month
        sim_dicti['state'] = np.append(sim_dicti['state'], np.repeat(state, len(y_art)))
        sim_dicti['year'] = np.append(sim_dicti['year'], year_arr)
        sim_dicti['month'] = np.append(sim_dicti['month'], month_arr)
        sim_dicti['y_art'] = np.append(sim_dicti['y_art'], y_art)
    df_sim = pd.DataFrame(sim_dicti)

    # DATA SAMPLING
    date_today = dt.datetime.today().strftime('%Y-%m-%d')
    DA_PATH = RESULT_PATH / 'data_sampling' / f"{n_samples}_samples" / date_today
    DA_PATH.mkdir(parents=True, exist_ok=True)
    da_filename = f"prescriptions_daily_sampled.bin"
    xar = sample_dataset(df=df_raw, da_path=DA_PATH,  da_filename=da_filename,
                         start=start, end=end,
                         df_ART=df_sim, times=n_samples, n_nodes=n_nodes)
    pickle.dump(xar, (DA_PATH / da_filename).open('wb'))

    if save_in_data_folder:
        pickle.dump(xar, (DATA_PATH / da_filename).open('wb'))


if __name__ == '__main__':
    main()