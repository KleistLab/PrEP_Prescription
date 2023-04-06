import colorcet as cc
import copy
import pickle

from src.models.models import Model, model_extended, c_intermitted, c_shi
from src.optimization.simulate import Simulator
from src.utils.plotting import *
from src.utils.stats import *
from src import DATA_PATH, RESULT_PATH, FIGURE_PATH, int_to_state, state_to_int


palette = sns.color_palette(cc.glasbey_bw_minc_20, n_colors=25)
sns.set_style('ticks')

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000


# def main():
# read sampled parameters
parameter_file_sampled = 'model_parameters_sampled.tsv'
parameter_df_sampled = pd.read_csv(DATA_PATH / parameter_file_sampled, sep='\t')
parameter_df_sampled['sim_endpoints'] = parameter_df_sampled.sim_endpoints.apply(
    lambda x: [int(i) for i in x[1:-1].split(',')])  # convert string to list[int] object

# read parameters from single file
parameter_file = 'model_parameters.tsv'
parameter_df = pd.read_csv(DATA_PATH / parameter_file, sep='\t')
parameter_df['sim_endpoints'] = parameter_df.sim_endpoints.apply(
    lambda x: [int(i) for i in x[1:-1].split(',')])  # convert string to list[int] object

# read sampled data
da_file = DATA_PATH / 'prescriptions_daily_sampled.bin'
da = pickle.load(da_file.open('rb'))

# read raw data
df_raw = pd.read_csv(DATA_PATH / 'prescriptions_daily.tsv', sep='\t')
df_raw = df_raw[df_raw['year'] >= 2017]

# read Number of people in need of PrEP
df_msm = pd.read_csv(DATA_PATH / 'population_at_risk.tsv', sep='\t')
in_need = {df_msm.iloc[i]['state']: df_msm.iloc[i]['total'] for i in range(len(df_msm))}


# SIMULATE MODELS FOR EACH STATE

# initiate Model and Simulator objects
model = Model(model_extended, fids=['f_art', 'f_prep'], pids=['k_art', 'k_prep', 'n_msm'])
sim = Simulator(model)

# create xarray storing all simulations
sim_end = 15000
states = sorted(pd.unique(parameter_df_sampled['state']))
dims = ['state', 'quantile', 'y', 't']
state_coords = states
sample_coords = sorted(pd.unique(parameter_df_sampled['sample']))
y_coords = ['y_art', 'y_prep', 'y_tot']
quantile_coords = [0.025, 0.25, 0.5, 0.75, 0.975]
t_coords = np.arange(sim_end + 1)   # simulating t timesteps results in t+1 timepoints (startpoint 0 + t)
arrays = np.zeros(shape=(len(states), len(quantile_coords), len(y_coords), sim_end+1))

for i_state, state in enumerate(states):
    print(f"--- Simulating {state}")
    df_parameters_state = parameter_df_sampled[parameter_df_sampled['state'] == state]
    state_array = np.zeros(shape=(len(sample_coords), len(y_coords), sim_end+1))
    for i_smp, (_, row) in enumerate(df_parameters_state.iterrows()):
        if row['success']:
            endpoints = copy.deepcopy(row['sim_endpoints'])
            i_end = endpoints[-1]
            endpoints[-1] = sim_end   # simulate for a long time
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
            state_array[i_smp, :] = np.nan     # dont simulate if parameter optimization was not successful
    state_array[state_array < 0] = 0    # set all negative values to 0
    state_quantiles = np.nanquantile(state_array, q=quantile_coords, axis=0)
    arrays[i_state] = state_quantiles
sim_xr = xr.DataArray(arrays,
                      coords=[state_coords, quantile_coords, y_coords, t_coords],
                      dims=dims)
print('sim_done')

# Plotting
start_date = dt.date(2017, 1, 1) + dt.timedelta(days=90)
end_date = start_date + dt.timedelta(days=parameter_df_sampled['sim_endpoints'][0][-1])
end_date_prediction = dt.date(2026, 1, 1)
end_date_milestones = dt.date(2050, 1, 1)

fig_extensions = ['svg', 'png']
for extension in fig_extensions:
    figpath_data = FIGURE_PATH / f'data_sampled.{extension}'
    figpath_simulation = FIGURE_PATH / f'model_simulations_sampled.{extension}'
    figpath_cov = FIGURE_PATH / f'coverage.{extension}'
    figpath_cov_prediction = FIGURE_PATH / f'coverage_prediction.{extension}'
    figpath_lockdown1 = FIGURE_PATH / f'lockdown_effect1.{extension}'
    figpath_lockdown2 = FIGURE_PATH / f'lockdown_effect2.{extension}'

    plot_data_sampled(da, df_raw, figpath_data)
    plot_state_simulations_sampled(sim_xr=sim_xr, start_date=start_date, end_date=end_date,
                                   df_ref=df_raw, figpath=figpath_simulation)
    plot_coverage(sim_xr=sim_xr, dicti_in_need=in_need, coeff_intermitted=c_intermitted, coeff_shi=c_shi,
                  start_date_sim=start_date, end_date_data=end_date, end_date_pred=end_date,
                  figpath=figpath_cov, plot_data_end=False)
    plot_coverage(sim_xr=sim_xr, dicti_in_need=in_need, coeff_intermitted=c_intermitted, coeff_shi=c_shi,
                  start_date_sim=start_date, end_date_data=end_date, end_date_pred=end_date_prediction,
                  figpath=figpath_cov_prediction, plot_data_end=True)
    plot_lockdown_effect(df_parameters=parameter_df_sampled, figpaths=[figpath_lockdown1, figpath_lockdown2])


filepath_milestones = RESULT_PATH / 'milestones.tsv'
compute_milestones(sim_xr, dicti_in_need=in_need, coeff_intermitted=c_intermitted, coeff_shi=c_shi,
                   start_date_sim=start_date, end_date=end_date_milestones,
                   coverage=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,  0.8, 0.9],
                   filepath=filepath_milestones)


# main()