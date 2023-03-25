import copy
import gc
import pickle

import numpy as np
import pandas as pd
import xarray as xr

import calendar
import datetime as dt
from datetime import datetime
import ray
import time

from scipy.stats import binom
from numba import jit

from src import int_to_state, state_to_int, DATA_PATH
from src import timeit, states_to_remove


def sample_dataset(df, df_ART, da_path, da_filename, times=1, n_nodes=5):
    """ Samples numbers of 30- and 90-pill prescriptions from binomial distribution and creates daily dataset from that.
    The function consists of three steps:
      1) Compute probabilities of 30- and 90-pill prescriptions for each month from provided dataset
      2) Sample the monthly 30- and 90-pill prescriptions from a binomial distribution, using the previously computed probabilities
      3) From the monthly prescriptions, daily prescriptions are computed by incorporating information about the package size:
         - Create an array containing zeros, where each entry corresponds to a day between 2018-01-01 and 2022-12-31
         - For each month check number of prescribed TDF/FTC packages and their size
         - For each prescription of size n draw a random day in the month, assuming the prescription was redeemed that day.
         - Starting from that day, increment the next n entries by 1

    :param df: Dataframe containing number of 30- and 90-pill prescriptions
    :param df_ART: Dataframe containing the number of TDF/FTC prescriptions/users for HIV Therapy
    :param da_path: Folder in which temporary data-arrays will be stored. Data arrays will be merged in the end
    :param times: Number of samples to create
    :param n_nodes:
    :return:
    """

    # compute number of people in need of TDF/FTC per month
    # We assume the number of people in need of TDF/FTC = number_in_need_of_PrEP + number_of_TDF/FTC_users_for_ART
    df_msm = pd.read_csv(DATA_PATH / 'msm_population.tsv', sep='\t')
    df_ART['total_pop'] = df_ART.apply(lambda x: x['y_art'] + df_msm[df_msm['state'] == x['state']]['total'].values[0], axis=1)
    df_ART.sort_values(by=['state', 'year', 'month'], inplace=True)
    df_ART.reset_index(inplace=True)

    # remove data from before 2017
    df = df[df.year >= 2017]

    # change structure of dataframe and sort
    ## Change from having a column reporting package size, to having a column for each package size containing the number of prescriptions
    df = df.groupby(by=['state', 'year', 'month', 'SE']).sum().reset_index('SE').\
        pivot(columns=['SE'], values='Verordnungen').reset_index()
    df.replace(np.nan, 0, inplace=True)
    df['30'] = df[28] + df[30] + df[35]
    df['90'] = df[84] + df[90]
    df = df[['state', 'year', 'month', '30', '90']]

    # remove some states (e.g. Berlin, Brandenburg, unidentified, ...) and change state_names from int to string
    df.drop(df[df['state'].isin(states_to_remove)].index, axis=0, inplace=True)
    df['state'] = df.apply(lambda x: int_to_state[int(x['state'])], axis=1)
    df.sort_values(['state', 'year', 'month'], inplace=True)
    df.reset_index(inplace=True)

    # add column containing number of msm per state
    df['total_population'] = df_ART['total_pop']

    # create coordinates for xarrays
    state_coords = list(df.state.unique())
    year_coords = df.year.unique().astype(int)
    month_coords = df.month.unique().astype(int)
    day_coords = np.arange(1, 32)
    size_coords = np.arange(1, times+1)
    pkg_size_coords = ['p30', 'p90']

    # get number of processed states and years
    n_states = len(state_coords)
    n_years = len(year_coords)
    n_months = 12

    # compute probabilities for 30- and 90-pill packages and sample n times from binominal distribution
    ar = np.zeros((times, 2, n_states, n_years, n_months))      # create empty array
    for _, row in df.iterrows():
        # compute array indices
        i_state = state_coords.index(row.state)
        i_year = int(row.year - min(year_coords))
        i_month = int(row.month - 1)

        # compute probabilities and total number of prescriptions
        n_prescriptions = row['30'] + row['90']
        pop = row['total_population']
        p30 = row['30'] / n_prescriptions   # probability that someone who takes PrEP gets a 30-pill package
        p30_tot = row['30'] / pop         # probability that someone in the risk group gets a 30-pill package (no matter wheter they take PrEP or not)
        p90 = row['90'] / n_prescriptions
        p90_tot = row['90'] / pop

        # sample from binominal distribution
        n30, n90 = sample_prep_users_2step(pop, p30, p90, p30_tot, p90_tot, times)    # sample from binominal distribution

        # add prescription numbers to array
        for k, (n3, n9) in enumerate(zip(n30, n90)):
            ar[k, 0, i_state, i_year, i_month] = n3
            ar[k, 1, i_state, i_year, i_month] = n9

    # create xarray
    xar = xr.DataArray(ar,
                       coords=[size_coords, pkg_size_coords, state_coords, year_coords, month_coords],
                       dims=['sample', 'pkg_size', 'state', 'year', 'month'])

    # create daily dataset
    date_to_idx = {date: k for k, date in enumerate(np.arange(dt.date(2017, 1, 1), dt.date(2022, 12, 31)).astype(dt.date))}
    idx_to_date = {value: key for key, value in date_to_idx.items()}
    arrays = []
    for state in state_coords:
        print(f'--- Processing {state}')
        state_ar = xar.sel(state=state)
        ray.init(num_cpus=n_nodes, ignore_reinit_error=True)
        out = []
        for _, sample_ar in state_ar.groupby('sample'):
            sample_df = sample_ar.to_dataframe(name=1).reset_index()
            sample_df.replace({'p30': 30, 'p90': 90}, inplace=True)
            sample_df.rename({'pkg_size': 'SE', 1: 'Verordnungen'}, inplace=True, axis=1)
            out.append(fill_array_remote.remote(sample_df, date_to_idx, idx_to_date))
        state_array = ray.get(out)
        ray.shutdown()
        da_state = xr.DataArray([state_array],
                                coords=[[state], size_coords, year_coords, month_coords, day_coords],
                                dims=['state', 'sample', 'year', 'month', 'day'])
        pickle.dump(da_state, (da_path / f"{state}_tmp.bin").open('wb'))

    print('--- Construct DataArray')
    data_arrays = sorted(da_path.glob('*tmp.bin'))
    for k, da_file in enumerate(data_arrays):
        da = pickle.load(da_file.open('rb'))
        if k == 0:
            da_merged = da
        else:
            da_merged = xr.concat([da_merged, da], dim='state')
    pickle.dump(da_merged, (da_path / da_filename).open('wb'))

    # remove temporary files
    for da_file in data_arrays:
        da_file.unlink(missing_ok=True)

    return da_merged

def create_datearray(shape, start_date, end_date=None):
    """ Creates a 3D zero-array with dimensions year-month-day
        All inner arrays have length 31. Entries representing non-existing dates (e.g. Feb 31st) are set to NaN
    """
    date_array = np.zeros(shape)
    start_year = start_date.year
    start_month = start_date.month
    year = start_year
    for year_array in date_array:
        month = start_month
        for month_array in year_array:
            for i in range(len(month_array)):
                day = i+1
                try:
                    date = dt.date(year, month, day)
                    if end_date:
                        if date > end_date:
                            date_array[year - start_year, month - 1, day - 1] = np.nan
                except:
                    date_array[year - start_year, month-1, day-1] = np.nan
            month += 1
        year += 1
    return date_array

def fill_array(df_state, start, end):
    """ From the monthly prescriptions, daily prescriptions are computed by incorporating information about the package size:
         - Create an array containing zeros, where each entry corresponds to a day between 2018-01-01 and 2022-12-31
         - For each month check number of prescribed TDF/FTC packages and their size
         - For each prescription of size n draw a random day in the month, assuming the prescription was redeemed that day.
         - Starting from that day, increment the next n entries by 1
    """
    start_year, start_month, start_day = start
    end_year, end_month, end_day = end
    start_date = dt.date(start_year, start_month, start_day)
    end_date = dt.date(end_year, end_month, end_day)
    date_to_idx = {date: k for k, date in enumerate(np.arange(start_date, end_date + dt.timedelta(days=2)).astype(dt.date))}
    idx_to_date = {value: key for key, value in date_to_idx.items()}
    state_array = create_datearray((end_year - start_year + 1, 12, 31),
                                   start_date=start_date,
                                   end_date=end_date)
    n_dates = len(date_to_idx)
    counts = np.zeros(n_dates)
    for _, row in df_state.iterrows():
        year = row.year
        month = row.month
        pkg_size = row.SE
        prescriptions = int(row['Verordnungen'])
        start_dates_prescriptions = randomdates(year, month, prescriptions)
        for date in start_dates_prescriptions:
            end_date_prescription = date + dt.timedelta(days=int(pkg_size))
            if end_date_prescription > end_date:
                end_date_prescription = end_date + dt.timedelta(days=1)
            i_start, i_end = date_to_idx[date], date_to_idx[end_date_prescription]
            counts[i_start:i_end] += 1

    for k, count in enumerate(counts):
        date = idx_to_date[k]
        if date.year > end_year:
            break
        i = date.year - start_year
        j = date.month - 1
        k = date.day - 1
        state_array[i, j, k] += count

    return state_array

@ray.remote
def fill_array_remote(df_state, date_to_idx, idx_to_date):
    return fill_array(df_state, date_to_idx, idx_to_date)

def randomdates(year, month, size):
    """ Creates an array of random dates in a given month """
    dates = calendar.Calendar().itermonthdates(year, month)
    return np.random.choice([date for date in dates if date.month == month], size=size)

def sample_prep_users_2step(n_msm, p30, p90, p30_tot, p90_tot, size):
    """ Samples number of PrEP users
            1) Sample number of PrEP users
            2) Sample number of 30- and 90-pill prescriptions
    """

    # Step1: sample number of PrEP users
    p_prep_user = p30_tot + p90_tot
    b_prep_users = binom(int(n_msm), p_prep_user)
    n_prep_users = b_prep_users.rvs(size)

    # Step2: Sample number of 30- and 90-pill prescriptions
    n30 = binom(n_prep_users, p30).rvs(size)
    n90 = n_prep_users - n30

    return n30, n90



