import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from src import state_to_int, RESULT_PATH
from src.utils.stats import compute_kprep_pvalue

sns.set_style('ticks')
FIGURE_PATH = RESULT_PATH / 'figures'

def plot_data_sampled(da_samples: xr.DataArray, df_raw, figpath=None):
    """ Plot sampled data. Plots the median, and 2.5, 25, 75 and 97.5 quantiles"""

    states = da_samples.coords['state'].to_numpy()

    # compute 2.5, 25, 50, 75 and 97.5 percentiles
    quantiles_to_compute = [0.025, 0.25, 0.5, 0.75, 0.975]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 10))
    k = 0
    for state in states:
        state_int = str(state_to_int[state])

        # get figure ax
        i = int(k / 4)
        j = np.mod(k, 4)
        ax = axes[i][j]
        k += 1

        # select state array
        da_state = da_samples.sel(state=state)
        quantiles = da_state.quantile(q=quantiles_to_compute, dim='sample')

        # convert dataarray to dataframe
        df_quantile = quantiles.to_dataframe('value')
        c = df_quantile.index.names
        df_quantile = df_quantile.reset_index(level=c).dropna(axis=0).set_index(c)  # remove NaN (i.e. invalid dates)

        # create date column
        df_quantile['date'] = df_quantile.reset_index(level=['year', 'month', 'day']).apply(
            lambda x: dt.date(int(x.year), int(x.month), int(x.day)), axis=1).values
        df_quantile.reset_index(level=['year', 'month', 'day'], drop=True, inplace=True)    # drop index columns referring to date

        t = df_quantile.loc[0.5].date
        t_raw = np.arange(dt.date(2017, 1, 1), dt.date(2022, 1, 1)).astype(dt.date)
        y_raw = df_raw[state_int].to_numpy()
        median = df_quantile.loc[0.5].value
        q_025 = df_quantile.loc[0.025].value
        q_25 = df_quantile.loc[0.25].value
        q_75 = df_quantile.loc[0.75].value
        q_975 = df_quantile.loc[0.975].value
        ax.plot(t, median, color='tab:blue')
        ax.plot(t_raw, y_raw, color='black', alpha=0.7, linestyle=':')
        ax.fill_between(t, q_25, q_75, color='tab:blue', alpha=0.2)
        ax.fill_between(t, q_025, q_975, color='tab:blue', alpha=0.1)
        ax.set_title(state)
    fig.tight_layout()
    fig.show()

    if figpath is None:
        figpath = FIGURE_PATH / 'data_sampled.png'
    fig.savefig(figpath, dpi=300)

def plot_state_simulations_sampled(sim_xr:xr.DataArray, start_date, end_date, df_ref, figpath=None):
    """ Creates a figure containing model simulations for each federal state, as well as whole Germany

    :param sim_xr: DataArray containing precomputed quantiles of simulation results.
    :param start_date: Date at which the simulation starts. Datetime.Date object
    :param end_date: Date at which the simulation ends. Datetime.Date object
    :param df_ref:  Dataframe containing reference data
    :param figpath: Filepath to which the figure will be saved
    :return:
    """
    date_range_ref = df_ref.apply(lambda x: dt.date(x.year, x.month, x.day), axis=1).to_numpy()
    date_range_sim = np.arange(start_date, end_date).astype(dt.date)
    i_end = len(date_range_sim)

    states = list(sim_xr.coords['state'].to_series())
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 10.6    ))
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            k = 4*i + j
            state = states[k]
            state_int = str(state_to_int[state])

            # plot data
            data_daily = df_ref[state_int].to_numpy()
            ax.plot(date_range_ref, data_daily, color='black', linestyle='--', label='Data', linewidth=1, alpha=0.7)

            # plot median
            ax.plot(date_range_sim,
                    sim_xr.sel(state=state, quantile=0.5, y='y_art', t=range(i_end)),
                    color='tab:orange', label='ART', linewidth=2, alpha=0.7)
            ax.plot(date_range_sim,
                    sim_xr.sel(state=state, quantile=0.5, y='y_prep', t=range(i_end)),
                    color='tab:green', label='PrEP', linewidth=2, alpha=0.7)
            ax.plot(date_range_sim,
                    sim_xr.sel(state=state, quantile=0.5, y='y_tot', t=range(i_end)),
                    color='tab:blue', label='Total', linewidth=2, alpha=0.7)

            # plot 0.25 - 0.75 quantiles
            ax.fill_between(date_range_sim,
                            sim_xr.sel(state=state, quantile=0.25, y='y_art', t=range(i_end)),
                            sim_xr.sel(state=state, quantile=0.75, y='y_art', t=range(i_end)),
                            color='tab:orange', alpha=0.2)
            ax.fill_between(date_range_sim,
                            sim_xr.sel(state=state, quantile=0.25, y='y_prep', t=range(i_end)),
                            sim_xr.sel(state=state, quantile=0.75, y='y_prep', t=range(i_end)),
                            color='tab:green', alpha=0.2)
            ax.fill_between(date_range_sim,
                            sim_xr.sel(state=state, quantile=0.25, y='y_tot', t=range(i_end)),
                            sim_xr.sel(state=state, quantile=0.75, y='y_tot', t=range(i_end)),
                            color='tab:blue', alpha=0.2)

            # plot 0.025 - 0.975 quantile
            ax.fill_between(date_range_sim,
                            sim_xr.sel(state=state, quantile=0.025, y='y_art', t=range(i_end)),
                            sim_xr.sel(state=state, quantile=0.975, y='y_art', t=range(i_end)),
                            color='tab:orange', alpha=0.1)
            ax.fill_between(date_range_sim,
                            sim_xr.sel(state=state, quantile=0.025, y='y_prep', t=range(i_end)),
                            sim_xr.sel(state=state, quantile=0.975, y='y_prep', t=range(i_end)),
                            color='tab:green', alpha=0.1)
            ax.fill_between(date_range_sim,
                            sim_xr.sel(state=state, quantile=0.025, y='y_tot', t=range(i_end)),
                            sim_xr.sel(state=state, quantile=0.975, y='y_tot', t=range(i_end)),
                            color='tab:blue', alpha=0.1)

            ax.set_title(state)
            if j == 0:
                ax.set_ylabel("# Prescriptions")
            if i == 3: # or (i == 2 and j == 3):
                ax.tick_params(axis='x', labelrotation=45, right=True)
                # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
                # ax.set_xticklabels(["01-2018", "01-2019", "01-2020", "01-2021"], rotation=45, ha='right')
                # ax.set_xticks([0, 12, 24, 36])
            else:
                ax.set_xticklabels([])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0, 0.93, 1, 0), loc='lower center', ncol=4, prop={'size': 18})
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.show()

    if figpath is None:
        figpath = FIGURE_PATH / 'model_simulations_sampled.png'
    fig.savefig(figpath, dpi=300)


def plot_coverage(sim_xr: xr.DataArray, dicti_in_need, coeff_intermitted, coeff_shi, start_date_sim, end_date_data, end_date_pred,
                  figpath=None, plot_data_end=False):
    """ Computes and plots the relative number of PrEP users over time in each federal state, as well as whole Germany.

    :param sim_xr: DataArray containing precomputed quantiles of simulation results.
    :param dicti_at_risk: Dictionary containing the number of people in need of PrEP for each federal state
    :param coeff_intermitted: Coefficient to correct for intermitted/on-demand use of PrEP
    :param coeff_shi: Coefficient to correct for self-payer and privatly insured PrEP users
    :param start_date_sim: Start date of the simulation
    :param end_date_data: End date of the data
    :param end_date_pred: End date of the prediction. Coverage will be plotted up to this point
    :param figpath: Filepath to save figure
    :param plot_data_end: If True, a vertical line will be plotted where the data ends and prediction starts
    :return:
    """

    if end_date_data >= end_date_pred:
        plot_data_end = False

    states = sorted(sim_xr.coords['state'].to_series())

    # create array containing dates
    start_date_plot = dt.date(2019, 9, 1)          # first date we actually want to plot (1st Sep 2019)
    date_range_1 = [start_date_plot + dt.timedelta(days=i) for i in range(0, (end_date_data - start_date_plot).days)]
    date_range_2 = [end_date_data + dt.timedelta(days=i) for i in range(0, (end_date_pred - end_date_data).days)]
    t_start = (start_date_plot - start_date_sim).days
    t_end_data = (end_date_data - start_date_sim).days
    t_end = (end_date_pred - start_date_sim).days

    # select dates to plot, estimate number of PrEP users and compute quantiles
    da_sim = sim_xr.sel(t=range(t_start, t_end), y='y_prep')     # assumes that the simulation starts at <start_date
    da_sim = da_sim * coeff_intermitted * coeff_shi   # estimate prep_users from prep_prescriptions

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 10))
    for k, state in enumerate(states):
        n_in_need = dicti_in_need[state]
        # da_state = da_quantiles.sel(state=state, y='y_prep') / n_in_need
        # q025, q25, q50, q75, q975 = da_state.sel(quantile=[0.025, 0.25, 0.5, 0.75, 0.975])

        i = int(k/4)
        j = np.mod(k, 4)
        ax = axes[i][j]
        ax.plot(date_range_1,
                da_sim.sel(state=state, quantile=0.5, t=range(t_start, t_end_data)).data / n_in_need,
                color='black', linewidth=1.5)
        ax.fill_between(date_range_1,
                        da_sim.sel(state=state, quantile=0.25, t=range(t_start, t_end_data)).data / n_in_need,
                        da_sim.sel(state=state, quantile=0.75, t=range(t_start, t_end_data)).data / n_in_need,
                        color='black', alpha=0.3)
        ax.fill_between(date_range_1,
                        da_sim.sel(state=state, quantile=0.025, t=range(t_start, t_end_data)).data / n_in_need,
                        da_sim.sel(state=state, quantile=0.975, t=range(t_start, t_end_data)).data / n_in_need,
                        color='black', alpha=0.15)

        ax.plot(date_range_2,
                da_sim.sel(state=state, quantile=0.5, t=range(t_end_data, t_end)).data / n_in_need,
                color='black', linewidth=1.5, linestyle='--')
        ax.fill_between(date_range_2,
                        da_sim.sel(state=state, quantile=0.25, t=range(t_end_data, t_end)).data / n_in_need,
                        da_sim.sel(state=state, quantile=0.75, t=range(t_end_data, t_end)).data / n_in_need,
                        color='black', linestyle='--', alpha=0.3)
        ax.fill_between(date_range_2,
                        da_sim.sel(state=state, quantile=0.025, t=range(t_end_data, t_end)).data / n_in_need,
                        da_sim.sel(state=state, quantile=0.975, t=range(t_end_data, t_end)).data / n_in_need,
                        color='black', linestyle='--', alpha=0.15)

        #ax.plot(t[idx_start:], y_prep_rel[idx_start:])
        # ax.axhline(0.1, linestyle='--', color='tab:red', alpha=0.5)
        # ax.axhline(0.3, linestyle='--', color='tab:red', alpha=0.5)
        # ax.axhline(0.5, linestyle='--', color='tab:red', alpha=0.5)
        # ax.axhline(0.8, linestyle='--', color='tab:red', alpha=0.5)
        if plot_data_end:
            ax.axvline(end_date_data, linestyle='--', color='tab:grey', alpha=0.8)
        ax.set_title(state)
        ax.set_ylim(top=1.02, bottom=-0.02)
        # if j != 0:
        #     ax.set_yticklabels([])
        # if i != 3:
        #     ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel('Coverage')
        k += 1
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.show()
    if figpath is None:
        figpath = FIGURE_PATH / 'coverage.png'
    fig.savefig(figpath, dpi=300)


def plot_lockdown_effect(df_parameters, figpaths, alpha1=0.05, alpha2=0.01):
    """ Plots the difference between the parameter values before and after the first (k_prep2, k_prep3) and second
    lockdown (k_prep4, k_prep5) as boxplot in two seperate figures. """
    df_parameters = df_parameters[df_parameters['success']]
    df_parameters.sort_values('state', inplace=True)
    states = pd.unique(df_parameters['state'])

    df_parameters['lockdown1_diff'] = df_parameters.apply(lambda x: x['k_prep3'] - x['k_prep2'], axis=1)
    df_parameters['lockdown2_diff'] = df_parameters.apply(lambda x: x['k_prep5'] - x['k_prep4'], axis=1)

    # compute pvalues for 1st lockdown
    print('\nP-Values (1st Lockdown)')
    pvalues = compute_kprep_pvalue(df_parameters, 'k_prep2', 'k_prep3')
    labels1 = []
    for state, p in zip(df_parameters['state'].unique(), pvalues):
        if p <= alpha2:
            label = f"{state} (**)"
        elif p <= alpha1:
            label = f"{state} (*)"
        else:
            label = state
        labels1.append(label)

    # compute pvalues for 2nd lockdown
    print('\nP-Values (1st Lockdown)')
    pvalues = compute_kprep_pvalue(df_parameters, 'k_prep4', 'k_prep5')
    labels2 = []
    for state, p in zip(df_parameters['state'].unique(), pvalues):
        if p <= alpha2:
            label = f"{state} (**)"
        elif p <= alpha1:
            label = f"{state} (*)"
        else:
            label = state
        labels2.append(label)

    fig1 = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_parameters, x='lockdown1_diff', y='state', orient='h', color='tab:grey', fliersize=1)
    plt.axvline(x=0, color='black', alpha=0.8)
    plt.xlabel(r'$k_{PrEP, 3} - k_{PrEP, 2}$')
    plt.ylabel(None)
    locs, _ = plt.yticks()
    plt.yticks(locs, labels1)
    fig1.tight_layout()
    fig1.show()

    fig2 = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_parameters, x='lockdown2_diff', y='state', orient='h', color='tab:grey', fliersize=1)
    plt.axvline(x=0, color='black', alpha=0.8)
    plt.xlabel(r'$k_{PrEP, 5} - k_{PrEP, 4}$')
    plt.ylabel(None)
    locs, _ = plt.yticks()
    plt.yticks(locs, labels2)
    fig2.tight_layout()
    fig2.show()

    for fig, figname in zip([fig1, fig2], figpaths):
        fig.savefig(figname, dpi=300)

    print('\nP-Values (1st Lockdown)')
    compute_kprep_pvalue(df_parameters, 'k_prep2', 'k_prep3')

    print('\nP-Values (2nd Lockdown)')
    compute_kprep_pvalue(df_parameters, 'k_prep4', 'k_prep5')

