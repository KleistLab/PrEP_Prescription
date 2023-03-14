import datetime as dt
import pandas as pd
import xarray as xr

from src import DATA_PATH, RESULT_PATH


def compute_kprep_pvalue(df_parameters, kprep_1, kprep_2):
    """ Compares two sets of parameters kprep_1 and kprep_2, and computes the p-value, under the assumption:
        H0: kprep_1 <= kprep_2
        H1: kprep_1 >  kprep_1

    :param df_parameters: Dataframe containing model parameters
    :param kprep_1: Columnname of first parameter
    :param kprep_2: Columnname of second parameter
    """

    df_parameters = df_parameters[df_parameters.success]
    pvalues = []
    for state in sorted(pd.unique(df_parameters['state'])):
        df_parameters_state = df_parameters[df_parameters['state'] == state]
        n_samples = len(df_parameters_state)
        pvalue = (df_parameters_state[kprep_1] > df_parameters_state[kprep_2]).sum() / n_samples
        pvalues.append(1-pvalue)
        print(f"State: {state}; p value: {1-pvalue}; {1-pvalue <= 0.05}")

    return pvalues


def compute_milestones(sim_xr: xr.DataArray, dicti_in_need, coeff_intermitted, coeff_shi, start_date_sim, end_date,
                       coverage=[0.1, 0.3, 0.5],
                       timepoints=[(2019, 12, 31), (2020, 6, 30), (2020, 12, 31), (2021, 6, 30), (2021, 12, 31),
                                   (2022, 12, 31), (2023, 12, 31), (2024, 12, 31), (2025, 12, 31),
                                   (2026, 12, 31), (2027, 12, 31), (2028, 12, 31), (2029, 12, 31), (2030, 12, 31)],
                       filepath=None):
    """ Computes the following stats:
        - Relative number of PrEP users (coverage) at given timepoints
        - Absolute number of PrEP users at given timepoints
        - Time at which certain coverages are reached

    :param sim_xr: DataArray containing simultion results (Dimensions: state-quantile-y-t)
    :param dicti_in_need: Dictionary with keys - Federal State, values - Number of people at risk
    :param coeff_intermitted: Coefficient to adjust for intermitted use. Used in computation from number of PrEP prescriptions to Users
    :param coeff_shi: Coefficient to adjust for self-payer/privatly insured. Used in computation from number of PrEP prescriptions to Users
    :param start_date_sim: Start date of the simulation. Date object.
    :param end_date: Last date to consider. Any later date will be set to end_date
    :param coverage: Coverages for whic
    :param timepoints: Timepoints at which coverages should be computed.
    :param filepath: Path to which results should be saved
    """

    # get number of people at risk for each state
    df_msm = pd.read_csv(DATA_PATH / 'msm_population.tsv', sep='\t')
    df_msm.sort_values(by='state', inplace=True)

    # compute prep users from prescriptions
    sim_xr = sim_xr * coeff_intermitted * coeff_shi

    df_dicti = {'state': []}
    for c in coverage:
        df_dicti[f'{int(c*100)}%'] = []
        df_dicti[f'{int(c*100)}%_CI_lb'] = []
        df_dicti[f'{int(c*100)}%_CI_ub'] = []

    endpoint = sim_xr.coords['t'].to_numpy()[-1] + 1
    date_range = [start_date_sim + dt.timedelta(days=i) for i in range(endpoint) if (start_date_sim + dt.timedelta(days=i)) <= end_date]

    date_indices = []
    dates = []
    for (y, m, d) in timepoints:
        date = dt.date(y, m, d)
        df_dicti[f"REL_COV_{date.strftime('%Y%m%d')}"] = []
        df_dicti[f"REL_COV_{date.strftime('%Y%m%d')}_CI_lb"] = []
        df_dicti[f"REL_COV_{date.strftime('%Y%m%d')}_CI_ub"] = []
        df_dicti[f"ABS_COV_{date.strftime('%Y%m%d')}"] = []
        df_dicti[f"ABS_COV_{date.strftime('%Y%m%d')}_CI_lb"] = []
        df_dicti[f"ABS_COV_{date.strftime('%Y%m%d')}_CI_ub"] = []
        date_index = (date - start_date_sim).days
        dates.append(date)
        date_indices.append(date_index)

    states = sim_xr.coords['state'].to_series()
    for state in states:
        n_in_need = dicti_in_need[state]
        da_state_abs = sim_xr.sel(state=state, quantile=[0.025, 0.5, 0.975], y='y_prep')
        da_state_rel = da_state_abs / n_in_need
        y_prep_q025, y_prep_q50, y_prep_q975 = da_state_rel.sel(quantile=[0.025, 0.5, 0.975])

        df_dicti['state'].append(state)

        # compute dates at which coverage c is reached
        coverage.sort()
        idx_ci_low = 0
        for c in coverage:
            dict_key = f'{int(c*100)}%'
            dict_key_lb = f"{dict_key}_CI_lb"
            dict_key_ub = f"{dict_key}_CI_ub"
            try:
                idx_ci_low = idx_ci_low + next(x[0] for x in enumerate(y_prep_q975[idx_ci_low:]) if x[1] >= c)
                date_ci_low = date_range[idx_ci_low]
            except:
                date_ci_low = date_range[-1]
            try:
                idx_med = idx_ci_low + next(x[0] for x in enumerate(y_prep_q50[idx_ci_low:]) if x[1] >= c)
                date_med = date_range[idx_med]
            except:
                date_med = date_range[-1]
            try:
                idx_ci_high = idx_med + next(x[0] for x in enumerate(y_prep_q025[idx_med:]) if x[1] >= c)
                date_ci_high = date_range[idx_ci_high]
            except:
                date_ci_high = date_range[-1]
            df_dicti[dict_key].append(date_med.strftime('%Y-%m-%d'))
            df_dicti[dict_key_lb].append((date_ci_low.strftime('%Y-%m-%d')))
            df_dicti[dict_key_ub].append((date_ci_high.strftime('%Y-%m-%d')))

        # compute coverage at date d
        for d, d_i in zip(dates, date_indices):
            df_dicti[f"REL_COV_{d.strftime('%Y%m%d')}"].append(y_prep_q50.data[d_i])
            df_dicti[f"REL_COV_{d.strftime('%Y%m%d')}_CI_lb"].append((y_prep_q025.data[d_i]))
            df_dicti[f"REL_COV_{d.strftime('%Y%m%d')}_CI_ub"].append((y_prep_q975.data[d_i]))
            df_dicti[f"ABS_COV_{d.strftime('%Y%m%d')}"].append(y_prep_q50.data[d_i] * n_in_need)
            df_dicti[f"ABS_COV_{d.strftime('%Y%m%d')}_CI_lb"].append(y_prep_q025.data[d_i] * n_in_need)
            df_dicti[f"ABS_COV_{d.strftime('%Y%m%d')}_CI_ub"].append(y_prep_q975.data[d_i] * n_in_need)

    df = pd.DataFrame(df_dicti)

    if filepath is None:
        filepath = RESULT_PATH / 'milestones.tsv'
    df.to_csv(filepath, sep='\t')