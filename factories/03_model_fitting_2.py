"""
Fits the 'extended' model to each state
Parameters are stored in a .tsv file. Parameter names are enumerated, corresponding to the simulation they're used in .
E.g. f_art0 -> f_art before PrEP; f_art1 -> f_art after PrEP; f_art2 -> f_art after 1st lockdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ray
import pickle
import datetime as dt

from scipy.signal import lfilter
from time import time

from src.models.models import Model, model_extended
from src.optimization.simulate import Simulator
from src.optimization.optimize import Parameter, OptimizationProblem
from src import DATA_PATH, RESULT_PATH, state_to_int, int_to_state


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


use_parameters = False  # Set to True to use the generated parameter file for subsequent steps
                        # Note: The file will be stored under data/model_parameters_sampled.tsv and replace any
                        #       previously generated file at that path!

def optimize_states(ref_data, state, fix_k, stop_indices, savepath, initial_guesses=None, optimization_kwargs={}):

    # get msm population size in state
    msm_df = pd.read_csv(DATA_PATH / 'population_at_risk.tsv', sep='\t')
    n_msm = msm_df[msm_df['state'] == state]['total'].values[0]

    # get initial guesses for initial values y0_art and y0_prep
    y0_art1 = ref_data[0]

    # create dictionary containing initial guesses
    if initial_guesses is None:
        sample_times = 10
        initial_guesses = {
            0: {'f_art': y0_art1, 'f_prep': 0, 'k_art': -1e-2, 'k_prep': 0, 'n_msm': n_msm},
            1: {'f_art': 1, 'f_prep': 0, 'k_art': 1, 'k_prep': 1e-4, 'n_msm': n_msm},
            2: {'f_art': 1, 'f_prep': 0, 'k_art': 1, 'k_prep': 1e-5, 'n_msm': n_msm},
            3: {'f_art': 1, 'f_prep': 0, 'k_art': 1, 'k_prep': -1e-5, 'n_msm': n_msm},
            4: {'f_art': 1, 'f_prep': 0, 'k_art': 1, 'k_prep': 1e-5, 'n_msm': n_msm},
            5: {'f_art': 1, 'f_prep': 0, 'k_art': 1, 'k_prep': -1e-5, 'n_msm': n_msm},
            6: {'f_art': 1, 'f_prep': 0, 'k_art': 1, 'k_prep': 1e-5, 'n_msm': n_msm},
        }
    else:
        sample_times = 0

    # initiate model and simulator
    model = Model(model_extended, fids=['f_art', 'f_prep'], pids=['k_art', 'k_prep', 'n_msm'])

    # initiate parameter objects for 1st simulation (no prep)
    p_y0_art1 = Parameter(pid='f_art', ptype='initial_value', sim_idx=[0], value=initial_guesses[0]['f_art'],
                          lower_bound=[y0_art1 - 0.5*y0_art1], upper_bound=[y0_art1 + y0_art1])
    p_k_art1 = Parameter(pid='k_art', ptype='parameter', fixed=True, sim_idx=[0],
                         value=initial_guesses[0]['k_art'], lower_bound=-1, upper_bound=1)
    p_y0_prep1 = Parameter(pid='f_prep', ptype='initial_value', sim_idx=0, value=initial_guesses[0]['f_prep'])
    p_k_prep1 = Parameter(pid='k_prep', ptype='parameter', value=initial_guesses[0]['k_prep'], sim_idx=0)
    p_n_msm1 = Parameter(pid='n_msm', ptype='parameter', value=initial_guesses[0]['n_msm'], sim_idx=0)
    p_beforePrep = [[p_y0_prep1, p_k_prep1, p_n_msm1], [p_y0_art1, p_k_art1]]

    # initiate parameter objects for 2nd optimization (with PrEP)
    parameters = [
        Parameter(pid='f_art', ptype='initial_value', sim_idx=[1, 2, 3, 4, 5, 6],
                  value=[initial_guesses[i]['f_art'] for i in [1, 2, 3, 4, 5, 6]]),
        Parameter(pid='k_art', ptype='parameter', fixed=True, sim_idx=[1, 2, 3, 4, 5, 6],
                  value=[initial_guesses[i]['k_art'] for i in [1, 2, 3, 4, 5, 6]]),
        Parameter(pid='f_prep', ptype='initial_value', sim_idx=[1, 2, 3, 4, 5, 6],
                  value=[initial_guesses[i]['f_prep'] for i in [1, 2, 3, 4, 5, 6]]),
        Parameter(pid='k_prep', ptype='parameter', sim_idx=[1],
                  value=[initial_guesses[i]['k_prep'] for i in [1]],
                  lower_bound=-0.1, upper_bound=0.1),     # Sep2019 - Dec2019
        Parameter(pid='k_prep', ptype='parameter', fixed=fix_k, sim_idx=[2, 4, 6],
                  value=[initial_guesses[i]['k_prep'] for i in [2, 4, 6]],
                  lower_bound=-0.1, upper_bound=0.1),
        Parameter(pid='k_prep', ptype='parameter', fixed=False, sim_idx=[3, 5],
                  value=[initial_guesses[i]['k_prep'] for i in [3, 5]],
                  lower_bound=-0.1, upper_bound=0.1),
        Parameter(pid='n_msm', ptype='parameter', sim_idx=[1, 2, 3, 4, 5, 6],
                  value=[initial_guesses[i]['n_msm'] for i in [1, 2, 3, 4, 5, 6]]),
    ]
    p_afterPrep = [[p for p in parameters if p.pid != 'k_prep'],
                   [p for p in parameters if p.pid == 'k_prep']]

    p_dicti, optimization_problem = optimize_model(
        model=model, data=ref_data, state=state, p_beforePrep=p_beforePrep, p_afterPrep=p_afterPrep,
        stop_indices=stop_indices, sample_times=sample_times, savepath=savepath, **optimization_kwargs
    )

    return p_dicti, optimization_problem

@ray.remote
def optimize_states_parallel(ref_data, state, fix_k, stop_indices, savepath, initial_guesses, optimization_kwargs):
    p_dicti, optimization_problem = optimize_states(ref_data, state, fix_k, stop_indices, savepath, initial_guesses, optimization_kwargs)
    return p_dicti, optimization_problem

def optimize_model(model, data, state, p_beforePrep, p_afterPrep, stop_indices, savepath, sample_times=20, **optimization_kwargs):
    """ Function to optimize unknowns for PrEP prediction
    This function is tailored specifically towards the PrEP dataset
    It first finds y0 and k_art for the y_ART function, by fitting the function against the first datapoints before Sep. 2019.
    In the next step all parameter values for y_PrEP are optimized.
    In the second step one can assume either no lockdown, only 1 lockdown (Apr. 2020) or two lockdowns (Apr. 2020 and Nov. 2020)
    """

    result_dicti = {i: {} for i in range(len(stop_indices))}

    # initialize simulator object
    sim = Simulator(model)

    ### 1st Optimization
    idx_prep = stop_indices[0]
    data_before_prep = data[:idx_prep]
    p_fixed, p_free = p_beforePrep

    # optimize model parameters
    opt = OptimizationProblem('optimize', sim, p_free, p_fixed, data_before_prep)
    opt.optimize(t_end=[idx_prep], t_step=1, smooth=False, daily=True, continuous=True, **optimization_kwargs)

    # add optimized parameters to results_dicti
    for pid in model.pids:
        result_dicti[0][pid] = opt.p['sim_0'][pid]
    for fid in model.fids:
        result_dicti[0][fid] = opt.y0['sim_0'][fid]

    ### 2nd Optimization
    ## Update Parameter objects
    p_fixed1, p_free1 = p_beforePrep
    p_fixed2, p_free2 = p_afterPrep
    for p in p_fixed2:
        if p.pid == 'k_art':            # 1. Pass optimal k_art value
            p.value = [result_dicti[0]['k_art'] for _ in p.value]
        if p.pid == 'f_art':            # 2. Pass initial f_art values
            p.value = [result_dicti[0]['f_art'] for _ in p.value]
    for p in p_free1:                   # 3. Update free parameters from 1st optimization with optimal values
        p.value = [result_dicti[0][p.pid]]

    p_fixed = p_fixed1 + p_free1 + p_fixed2     # all parameter values in sim_0 (before PrEP) are already fitted and therefore fixed
    p_free = p_free2

    # optimize model parameters
    if sample_times > 0:
        save_results = True
    else:
        save_results = False

    opt = OptimizationProblem('optimization', sim, p_free, p_fixed, data)
    opt.optimize(t_end=stop_indices, t_step=1, smooth=False, daily=True, continuous=True,
                 sample_parameters=['k_prep'], sample_times=sample_times, n_threads=10,
                 save_results=save_results,
                 savepath=savepath / f"optimization_{state}.tsv",
                 **optimization_kwargs)
    # add optimized parameters to results_dicti
    for sim_id in opt.p.keys():
        i = int(sim_id.split('_')[1])
        for pid, p in opt.p[sim_id].items():
            result_dicti[i][pid] = p
        for fid, y0 in opt.y0[sim_id].items():
            result_dicti[i][fid] = y0

    return result_dicti, opt


def main():

    # set hyperparameters for optimization
    optimization_kwargs = {
        'method': 'Nelder-Mead',
        'options': {'maxiter': 1000}
    }

    # create path to store results
    date_today = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    savepath = RESULT_PATH / 'optimization_results' / date_today
    savepath.mkdir(parents=True, exist_ok=True)

    # settings (fix k_prep, stop dates, start year)
    # NOTE: The sim_idx attribute in the parameter objects in optimize_states() have to be adjusted according to chosen stop_dates
    fix_k = False
    start_year = 2017
    years = np.arange(start_year, 2022)
    stop_dates = [
        '2019-08-31',       # start of insurance coverage
        '2019-11-30',       # end of initial increase
        '2020-03-31',       # 1st lockdown
        '2020-06-30',       # end of 1st lockdown decrease
        '2020-11-30',       # 2nd lockdown
        '2021-02-28',       # end of 2nd lockdown decrease
    ]

    # read data and drop some states
    xr_file = DATA_PATH / 'prescriptions_daily_sampled.bin'
    prescription_xr = pickle.load(xr_file.open('rb'))
    prescription_xr = prescription_xr.sel(year=years)
    state_ints = [1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 19, 20, 21]
    states = [int_to_state[i] for i in state_ints]

    # create stop indices based on stop_dates
    prescription_df_temp = prescription_xr.sel(sample=1, state='Baden-Württemberg').to_dataframe(name=1).reset_index()
    prescription_df_temp.dropna(axis=0, inplace=True)
    prescription_df_temp = prescription_df_temp.iloc[90:].reset_index(drop=True)     # remove first 90 days
    prescription_df_temp['date'] = prescription_df_temp.apply(
        lambda x: dt.date(x.year, x.month, x.day).strftime('%Y-%m-%d'), axis=1)
    stop_indices = [list(prescription_df_temp.date).index(stop_date) for stop_date in stop_dates] + [len(prescription_df_temp)]

    parameter_dicti = {}
    success_dicti = {}  # tracks whether the optimization was a success
    sim_dicti = {}
    for state in states:
        print(f"Optimizing {state}")
        state_xr = prescription_xr.sel(state=state)
        samples = state_xr.coords['sample'].to_numpy()

        # Sample initial parameter guesses to find a good initial guess
        # use mean of all samples as reference dataset
        print(f"--- Find good initial guess")
        ref_data = state_xr.median(dim='sample').to_dataframe(name='values').dropna(axis=0).reset_index()['values'].to_numpy()
        ref_data = ref_data[90:]    # remove first 90 days
        t = time()
        p_dicti, opt_problem = optimize_states(ref_data=ref_data, state=state, fix_k=fix_k,
                                               stop_indices=stop_indices, savepath=savepath,
                                               optimization_kwargs=optimization_kwargs)
        print(f"--- --- Initial Optimization done: {round((time() - t) / 60, 2)} [min]")

        # Fit model onto each data sample
        print(f"--- Fitting model onto sampled datasets")
        t = time()
        ray.init()
        out = []
        for sample in samples:
            ref_data = state_xr.sel(sample=sample).to_dataframe(name='values').dropna(axis=0).reset_index()['values'].to_numpy()
            ref_data = ref_data[90:]        # remove first 90 days
            out.append(optimize_states_parallel.remote(ref_data=ref_data, state=state, fix_k=fix_k,
                                                       stop_indices=stop_indices, savepath=savepath,
                                                       initial_guesses=p_dicti, optimization_kwargs=optimization_kwargs))
        fitting_results = ray.get(out)
        ray.shutdown()
        print(f"--- --- All model optimizations done: {round((time() - t)/60, 2)} [min]\n")

        p_dictis = [result[0] for result in fitting_results]
        opt_problems = [result[1] for result in fitting_results]

        parameter_dicti[state] = {}
        for rep, p_dicti in zip(samples, p_dictis):
            parameter_dicti[state][rep] = p_dicti
        success_dicti[state] = {}
        for rep, opt_problem in zip(samples, opt_problems):
            success_dicti[state][rep] = opt_problem.success
        # sim_results = opt_problem.simulate(smooth=False)
        # sim_dicti[state] = {'y': sim_results.y, 't': sim_results.t}

    opt_problem = opt_problems[0]
    sim_endpoints = opt_problem.timepoints
    sim_t_step = opt_problem.t_step

    # create dictionary with keys - <pid_simid> and values - list
    df_dicti = {'state': [], 'sample': [], 'success': [], 'sim_endpoints': [], 'sim_t_step': [],}
    unknowns = sorted([p.pid for p in opt_problem.free_parameters] + [p.pid for p in opt_problem.fixed_parameters])
    for unknown in unknowns:
        for i in range(opt_problem.n_sim):
            df_dicti[f"{unknown}{i}"] = []


    for state, state_dicti in parameter_dicti.items():
        for sample, sample_dicti in state_dicti.items():
            df_dicti['state'].append(state)
            df_dicti['sample'].append(sample)
            df_dicti['success'].append(success_dicti[state][sample])
            df_dicti['sim_endpoints'].append(sim_endpoints)
            df_dicti['sim_t_step'].append(sim_t_step)
            for i, p_dicti in sample_dicti.items():
                for pid, value in p_dicti.items():
                    df_dicti[f"{pid}{i}"].append(value)

    parameter_df = pd.DataFrame(df_dicti)
    savepath = RESULT_PATH / "model_parameters" / "model_parameters_sampled.tsv"
    parameter_df.to_csv(savepath, sep='\t', index=False)

    if use_parameters:
        parameter_df.to_csv(DATA_PATH / 'model_parameters_sampled.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
