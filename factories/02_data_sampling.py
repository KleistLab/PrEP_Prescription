import numpy as np
import pandas as pd
import xarray as xr

import datetime
import pickle
import calendar

from src.sampling.sampling import sample_dataset
from src.models import prep_model
from src.optimization.simulate import Simulator
from src import DATA_PATH, RESULT_PATH

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#TODO: ADD MODEL FITTING ONTO RAW DATA
#TODO: ADD PROPER FOLDER STRUCTURE

def main():

    # ESTIMATE NUMBER OF TDF/FTC PRESCRIPTIONS FOR ART
    ## initialize Model and Simulator
    model = prep_model
    sim = Simulator(model)

    ## get model parameters
    df_parameters = pd.read_csv(RESULT_PATH / 'model_parameters' / 'model_parameters_rawdata_20230222.tsv', sep='\t')
    df_parameters['sim_endpoints'] = df_parameters.sim_endpoints.apply(
        lambda x: [int(i) for i in x[1:-1].split(',')])  # convert string to list[int] object

    ## get indices of last days in month
    date_range = np.arange(datetime.date(2017, 1, 1), datetime.date(2022, 1, 1)).astype(datetime.date)    # all dates from 2017-1-1 to 2021-12-31
    last_days = [datetime.date(y, m, calendar.monthrange(y, m)[1]) for y in range(2017, 2022) for m in range(1, 13)]  # last day of each month within that timerange
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
    DA_PATH = DATA_PATH / 'data_sampling'
    da_filename = "prescriptions_daily_1000smp_20230306.bin"
    df_raw = pd.read_csv(DATA_PATH / 'prescriptions_raw.tsv', sep='\t')
    xar = sample_dataset(df=df_raw, da_path=DA_PATH, da_filename=da_filename,
                         df_ART=df_sim, times=1000, n_nodes=10)
    pickle.dump(xar, (DA_PATH / da_filename).open('wb'))


if __name__ == '__main__':
    main()