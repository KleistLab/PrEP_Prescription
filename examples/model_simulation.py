""" This script simulates the final model and stores the simulation results in an xarray.
The output of sim_final_model() can be used to create the final results (figures, tsv-file)
compute_additional_data() is a helper function that creates some additional objects needed for plotting.
The final few lines shows how to use these functions and plot everything
"""

import colorcet as cc
import copy
import pickle

from src.models.models import Model, model_extended, c_intermitted, c_shi
from src.optimization.simulate import Simulator
from src.utils.plotting import *
from src.utils.stats import *
from src import DATA_PATH, RESULT_PATH, FIGURE_PATH, int_to_state, state_to_int


parameter_file = DATA_PATH / 'model_parameters_sampled.tsv'
def sim_final_model(parameter_file=parameter_file, n_steps=20000):

    parameter_df_sampled = pd.read_csv(parameter_file, sep='\t')

    # initiate Model and Simulator objects
    model = Model(model_extended, fids=['f_art', 'f_prep'], pids=['k_art', 'k_prep', 'n_msm'])
    sim = Simulator(model)

    # create xarray storing all simulations
    sim_end = n_steps
    states = sorted(pd.unique(parameter_df_sampled['state']))
    dims = ['state', 'quantile', 'y', 't']
    state_coords = states
    sample_coords = sorted(pd.unique(parameter_df_sampled['sample']))
    y_coords = ['y_art', 'y_prep', 'y_tot']
    quantile_coords = [0.025, 0.25, 0.5, 0.75, 0.975]
    t_coords = np.arange(sim_end + 1)  # simulating t timesteps results in t+1 timepoints (startpoint 0 + t)
    arrays = np.zeros(shape=(len(states), len(quantile_coords), len(y_coords), sim_end + 1))

    for i_state, state in enumerate(states):
        print(f"--- Simulating {state}")
        df_parameters_state = parameter_df_sampled[parameter_df_sampled['state'] == state]
        state_array = np.zeros(shape=(len(sample_coords), len(y_coords), sim_end + 1))
        for i_smp, (_, row) in enumerate(df_parameters_state.iterrows()):
            if row['success']:
                endpoints = copy.deepcopy(row['sim_endpoints'])
                i_end = endpoints[-1]
                endpoints[-1] = sim_end  # simulate for a long time
                t_step = row['sim_t_step']
                y0 = []
                p = []
                for l in range(len(endpoints)):
                    y0.append([row[f"{fid}{l}"] for fid in model.fids])
                    p.append([row[f"{pid}{l}"] for pid in model.pids])
                sim_results = sim.simulate(endpoints, t_step, p, y0)
                state_array[i_smp, 0] = sim_results.y[0]
                state_array[i_smp, 1] = sim_results.y[1]
                state_array[i_smp, 2] = sim_results.y.sum(axis=0)
            else:
                state_array[i_smp, :] = np.nan  # dont simulate if parameter optimization was not successful
        state_array[state_array < 0] = 0  # set all negative values to 0
        state_quantiles = np.nanquantile(state_array, q=quantile_coords, axis=0)
        arrays[i_state] = state_quantiles

    sim_da = xr.DataArray(arrays,
                          coords=[state_coords, quantile_coords, y_coords, t_coords],
                          dims=dims)

    return sim_da

def compute_additional_data():
    """ Computes all kinds of additional data that is needed for plotting, such as 'start_date', 'end_date' or 'in_need_dicti'
    """

    parameter_df_sampled = pd.read_csv(parameter_file, sep='\t')
    start_date = dt.date(2017, 1, 1) + dt.timedelta(days=90)    # first timepoint in simulation (1st april 2017)
    end_date = start_date + dt.timedelta(days=parameter_df_sampled['sim_endpoints'][0][-1])     # last timepoint in data

    # read raw data
    df_raw = pd.read_csv(DATA_PATH / 'prescriptions_daily.tsv', sep='\t')
    df_raw = df_raw[df_raw['year'] >= 2017]

    # read Number of people in need of PrEP
    df_msm = pd.read_csv(DATA_PATH / 'population_at_risk.tsv', sep='\t')
    in_need = {df_msm.iloc[i]['state']: df_msm.iloc[i]['total'] for i in range(len(df_msm))}

    return df_raw, in_need, start_date, end_date

from src.utils.plotting import *
from src.models.models import c_intermitted, c_shi

sim_da = sim_final_model()
df_raw, in_need, start_date, end_date = compute_additional_data()
parameter_df_sampled = pd.read_csv(parameter_file, sep='\t')
end_date_prediction = dt.date(2026, 1, 1)
end_date_milestones = dt.date(2050, 1, 1)



plot_state_simulations_sampled(sim_xr=sim_da, start_date=start_date, end_date=end_date,
                               df_ref=df_raw)
plot_coverage(sim_xr=sim_da, dicti_in_need=in_need, coeff_intermitted=c_intermitted, coeff_shi=c_shi,
              start_date_sim=start_date, end_date_data=end_date, end_date_pred=end_date,
              plot_data_end=False)
plot_coverage(sim_xr=sim_da, dicti_in_need=in_need, coeff_intermitted=c_intermitted, coeff_shi=c_shi,
              start_date_sim=start_date, end_date_data=end_date, end_date_pred=end_date_prediction,
              plot_data_end=True)
plot_lockdown_effect(df_parameters=parameter_df_sampled)