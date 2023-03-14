import numpy as np
import itertools

import pandas as pd
import ray
import datetime

from typing import List, Collection, Union
from numbers import Number
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import loguniform

from src.optimization.simulate import Simulator


class Parameter:
    """ Class to store parameters, their bounds and initial values for fitting
    """

    def __init__(self, pid: str, ptype: str, value: Union[Collection[Number], Number],
                 sim_idx: Union[Collection[int], int], fixed: bool = False,
                 lower_bound: Union[Collection[Number], Number] = None,
                 upper_bound: Union[Collection[Number], Number] = None):
        """
        :param pid: Parameter ID as specified in the model
        :param ptype: 'parameter' or 'initial_value' for model parameter or initial function value, respectively
        :param value: List of parameter values, one per simulation. If the parameter will be optimized, the value is used as initial value
        :param sim_idx: List of indices that specify in which simulation the parameter is used (necessary since simulation might be restarted)
        :param fixed: Set to True if the same parameter value will be used in all simulations
        :param lower_bound: List of lower bounds, one per simulation. If not defined it will be set to -infinity
        :param upper_bound: List of upper bounds, one per simulation. If not defined it will be set to infinity
        """
        # FIXME: Make sure all provided lists have the same length (value, sim_idx, ub, lb)
        # FIXME: Make sure provided type is either 'parameter' or 'initial_value'
        self.pid = pid
        self.fixed = fixed
        self.__set_ptype(ptype)
        self.__set_sim_idx(sim_idx)
        self.__set_values(value)
        self.__set_bounds(lower_bound, upper_bound)

    def __set_ptype(self, ptype):
        if ptype not in ['parameter', 'initial_value']:
            raise ValueError("Attribute 'ptype' must be either 'parameter' or 'initial_value'")
        self.ptype = ptype

    def __set_sim_idx(self, sim_idx):
        if isinstance(sim_idx, Collection):
            self.sim_idx = list(sim_idx)
        elif isinstance(sim_idx, int):
            self.sim_idx = [sim_idx]
        else:
            raise ValueError(f"Attribute 'sim_idx' must be either of type int or Collection")
        self.n_sim = len(self.sim_idx)

    def __set_values(self, value):
        if isinstance(value, Collection):
            if len(value) != len(self.sim_idx):
                raise ValueError(f"Attributes sim_index and value must have the same length: {len(value)} != {len(self.sim_idx)}")
            self.value = list(value)
        elif isinstance(value, Number):
            self.value = [value] * len(self.sim_idx)
        else:
            raise ValueError(f"Attribute 'value' must be either of type Number or Collection")

    def __set_bounds(self, lower_bound, upper_bound):
        if lower_bound is None:
            self.lower_bound = [-np.inf] * self.n_sim
        elif isinstance(lower_bound, Collection):
            if len(lower_bound) != self.n_sim:
                raise ValueError(f"Attributes sim_index and value must have the same length: "
                                 f"{len(self.lower_bound)} != {self.n_sim}")
            else:
                self.lower_bound = list(lower_bound)
        elif isinstance(lower_bound, Number):
            self.lower_bound = [lower_bound] * self.n_sim
        else:
            raise ValueError(f"Attribute 'lower_bound' must be either of type Number or Collection")

        if upper_bound is None:
            self.upper_bound = [np.inf] * self.n_sim
        elif isinstance(upper_bound, Collection):
            if len(upper_bound) != self.n_sim:
                raise ValueError(f"Attributes sim_index and value must have the same length: "
                                 f"{len(self.upper_bound)} != {self.n_sim}")
            else:
                self.upper_bound = list(upper_bound)
        elif isinstance(upper_bound, Number):
            self.upper_bound = [upper_bound] * self.n_sim
        else:
            raise ValueError(f"Attribute 'upper_bound' must be either of type Number or Collection")

        for lb, ub in zip(self.lower_bound, self.upper_bound):
            if lb >= ub:
                raise ValueError(f"Lower bound must be less than upper bound: {lb} !< {ub}")


class OptimizationProblem:

    def __init__(self, opid: str, sim: Simulator, free_parameters: List[Parameter], fixed_parameters: List[Parameter],
                 y_ref: np.ndarray, obj_fun: str = 'default'):
        """
        :param opid: ID of optimization problem
        :param sim: Simulator object containing the model
        :param free_parameters: List of free parameters (to be optimized). Have to be Parameter objects
        :param fixed_parameters: List of fixed parameters (won't be optimized). Have to be Parameter objects
        :param y_ref: Reference data
        """
        self.opid = opid
        self.sim = sim
        self.free_parameters = free_parameters
        self.fixed_parameters = fixed_parameters
        self.y_ref = y_ref
        self.__set_obj_fun(obj_fun)       # NOTE: One could specify more objective functions, which then will be selected by an argument
        self.n_sim = self.__compute_nsim()
        self.__check_parameters()
        self.y0 = None
        self.p = None
        self.timepoints = None
        self.t_step = None
        self.continuous = False
        self.success = False

    def __obj_fun(self, parameters, t_end, t_step, smooth, daily, continuous=False):
        """ Objective function that uses least squares """
        y0, p = self.__map_parameters(parameters)
        try:
            if continuous:
                y_hat = self.sim.simulate_continuous(t_end, t_step, p, y0, smooth).y.sum(axis=0)
            else:
                y_hat = self.sim.simulate(t_end, t_step, p, y0, smooth).y.sum(axis=0)
            if daily:
                if continuous:  # we assume that in case of daily data & continuous simulation we use data with datapoints for each day
                    #y_diff = (y_hat - self.y_ref) ** 2
                    start = 0
                    y_diff = 0
                    for end in t_end:
                        y_diff += ((y_hat[start:end] - self.y_ref[start:end]) ** 2).sum(axis=0)/(end-start)
                        start = end
                    return y_diff
                else:
                    y_diff = (y_hat[2:] - self.y_ref[2:]) ** 2    # only remove the first two datapoints (2018-01 and 2018-02)
            else:
                y_diff = (y_hat - self.y_ref) ** 2
            return y_diff.sum(axis=0)
        except:
            # FIXME: This is a quick and dirty fix!
            # FIXME: objective function once threw an error in line 135 (daily=continous=True) Different length of arrays
            return np.inf

    def __map_parameters(self, parameters):
        """ Maps the parameters passed to the objective function to p and y0 that can be passed to solve_ivp
            Also fills up p and y0 with the fixed values

        solve_ivp() expects an array containing the initial values for each function of the ODE, as well as an array
        containing the parameter values of the ODE. Both these arrays must be in the correct order. Since sometimes
        only a subset of the parameters and/or initial values are fitted, a function is needed that fills up these
        arrays correctly. This is what this function is for.

        It takes the unknowns that need to be optimized and maps them onto the correct position in these arrays. This is
        done by looking at the ids of the free parameters and looking up their position in the fids/pids argument of the
        Model class. The remaining positions (fixed parameters) are filled up from the 'fixed_parameters' variable """

        # Create empty arrays with the correct shape
        y0 = np.zeros((self.n_sim, len(self.sim.model.fids)))
        y0[:] = np.nan
        p = np.zeros((self.n_sim, len(self.sim.model.pids)))
        p[:] = np.nan

        # map free parameters (unknowns) to p and y0
        k = 0   # index of the current unknown parameter that needs to be mapped
        for free_parameter in self.free_parameters:
            m = 0       # important if same parameter value is used for all simulations. Counts for how many simulations the value is being used
            pid = free_parameter.pid
            for i in free_parameter.sim_idx:
                if free_parameter.ptype == 'initial_value':
                    j = self.sim.model.fids.index(pid)
                    y0[i][j] = parameters[k]    # i-sim_idx; j-p_idx
                elif free_parameter.ptype == 'parameter':
                    j = self.sim.model.pids.index(pid)
                    p[i][j] = parameters[k]     # i-sim_idx; j-p_idx

                # update k. Either right now (if different values are used in the different simulations) or later (if same value is used)
                if free_parameter.fixed:
                    m += 1
                else:
                    k += 1
            k += m      # update k in case the parameter was 'fixed'

        # fill p and y0 up with fixed values
        for fixed_parameter in self.fixed_parameters:
            pid = fixed_parameter.pid
            for p_value, i in zip(fixed_parameter.value, fixed_parameter.sim_idx):
                if fixed_parameter.ptype == 'initial_value':
                    j = self.sim.model.fids.index(pid)
                    y0[i][j] = p_value
                elif fixed_parameter.ptype == 'parameter':
                    j = self.sim.model.pids.index(pid)
                    p[i][j] = p_value

        # make sure y0 and p were filled up (probably not needed since input is now checked during initialization)
        if np.any(np.isnan(y0)) or np.any(np.isnan(p)):
            raise ValueError("Not all parameters were mapped")

        return y0, p

    def __store_sim_parameters(self, parameters, timepoints, t_step, continuous=False):
        """Store all parameters (optimized and fixed) in self.y0 (initial values) and self.p (parameters)
        """
        y0_dicti = {}
        p_dicti = {}
        for i in range(self.n_sim):
            y0_dicti[f"sim_{i}"] = {}
            p_dicti[f"sim_{i}"] = {}

        k = 0
        for free_parameter in self.free_parameters:
            m = 0
            pid = free_parameter.pid
            for i in free_parameter.sim_idx:
                sim_key = f"sim_{i}"
                if free_parameter.ptype == 'initial_value':
                    y0_dicti[sim_key][pid] = parameters[k]
                elif free_parameter.ptype == 'parameter':
                    p_dicti[sim_key][pid] = parameters[k]
                if free_parameter.fixed:
                    m += 1
                else:
                    k += 1
            k += m

        for fixed_parameter in self.fixed_parameters:
            pid = fixed_parameter.pid
            for k, i in enumerate(fixed_parameter.sim_idx):
                sim_key = f"sim_{i}"
                if fixed_parameter.ptype == 'initial_value':
                    y0_dicti[sim_key][pid] = fixed_parameter.value[k]
                elif fixed_parameter.ptype == 'parameter':
                    p_dicti[sim_key][pid] = fixed_parameter.value[k]

        self.y0 = y0_dicti
        self.p = p_dicti
        self.timepoints = timepoints
        self.t_step = t_step


    def optimize(self, t_end, t_step, smooth=False, daily=False, continuous=False,
                 sample_parameters: List = [], sample_times=0, n_threads=1,
                 save_results=False, savepath=None, **kwargs):
        """ Fitting of optimization problem using scipy's minimize() function
        :param t_end: List containing all endpoints. Will be passed to Simulator object
        :param t_step: Step size for function evaluation. Will be passed to Simulator object
        :param smooth: Set to True if simulation in objective function should be smoothed (by moving average)
                       Makes sense if the reference data is smoothed as well
        """
        if len(t_end) != self.n_sim:
            raise ValueError(f"Number of simulation endpoints (t_end) is not equal to the number of simulation specified"
                             f"by the passed parameters: {len(t_end)} != {self.n_sim}")
        self.continuous = continuous
        # if no parameters to sample, don't sample at all
        if sample_parameters == []:
            sample_times = 0

        # Create two 1D arrays, one containing initial guesses of parameters, and the other one containing their bounds. (scipy.optimize.minimize expects a 1D-array)
        parameters = []
        p = []
        bounds = []
        for free_parameter in self.free_parameters:
            for p_value, lb, ub in zip(free_parameter.value, free_parameter.lower_bound, free_parameter.upper_bound):
                p.append(p_value)
                bounds.append((lb, ub))
        parameters.append(p)

        # sample parameters
        for _ in range(sample_times):
            p = []
            for free_parameter in self.free_parameters:
                for p_value, lb, ub in zip(free_parameter.value, free_parameter.lower_bound,
                                           free_parameter.upper_bound):
                    if free_parameter.pid in sample_parameters:
                        p_value = self.__sample_parameter(lb, ub)
                    p.append(p_value)
            parameters.append(p)

        # optimize unknown parameter values
        if n_threads == 1 or sample_times == 0:
            # single node version
            optimization_results = []
            for p in parameters:
                result = minimize(fun=self.fun, x0=p, bounds=bounds, args=(t_end, t_step, smooth, daily, continuous), **kwargs)
                optimization_results.append((p, result))
        else:
            # multithread version
            ray.init(num_cpus=n_threads, ignore_reinit_error=True)
            out = []
            for p in parameters:
                out.append((p, self.__minimize.remote(self, p, bounds, t_end, t_step, smooth, daily, continuous, kwargs)))
            optimization_results = [(p, ray.get(ray_object)) for p, ray_object in out]
            ray.shutdown()

        # get optimal parameters with from successful optimization
        optimization_results.sort(key=lambda x: x[1].fun)   # sort by objective value
        optimization_results.sort(key=lambda x: x[1].success, reverse=True)     # sort by optimization success
        _, p_opt = optimization_results[0]

        # set attribute 'success' to True if optimization resulting in the optimal parameters was a success
        self.success = p_opt.success

        # save optimization results (initial values, the resulting optimal parameters and the objective value)
        if save_results:
            self.__save_optimization_results(optimization_results, savepath)

        # store all parameters in object variable
        self.__store_sim_parameters(p_opt.x, t_end, t_step)

        # if continuous simulation was used, only y(t=0) matters. Other initial values are therefore not fitted
        # get other initial values from continuous simulation, so that a normal simulation with multiple steps can be run as well
        if continuous:
            sim_result = self.simulate(continuous=True)
            y0 = np.take(sim_result.y, [0] + t_end[:-1], axis=1)
            for k, fid in enumerate(self.sim.model.fids):
                for sim_id, dicti in self.y0.items():
                    i = int(sim_id.split('_')[1])
                    dicti[fid] = y0[k][i]
        return p_opt

    def simulate(self, smooth=False, continuous=False):
        y0 = []
        p = []
        for k, _ in enumerate(self.timepoints):
            sim_id = f"sim_{k}"
            y0.append([self.y0[sim_id][fid] for fid in self.sim.model.fids])
            p.append([self.p[sim_id][pid] for pid in self.sim.model.pids])
        if continuous:
            sim_result = self.sim.simulate_continuous(self.timepoints, self.t_step, parameters=p, y0=y0, smooth=smooth)
        else:
            sim_result = self.sim.simulate(self.timepoints, self.t_step, parameters=p, y0=y0, smooth=smooth)
        return sim_result


    ### HELPERS ###
    @ray.remote
    def __minimize(self, p, bounds, t_end, t_step, smooth, daily, continuous, kwargs):
        return minimize(fun=self.fun, x0=p, bounds=bounds, args=(t_end, t_step, smooth, daily, continuous), **kwargs)

    def __save_optimization_results(self, optimization_results, savepath):
        if savepath is None:
            RESULT_PATH = Path(__file__).parent.parent.parent / 'results'
            savepath = RESULT_PATH / f"parameter_optimization_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.tsv"
            print(f"No savepath for optimization results provided. Results will be stored under: {savepath}")

        df_dicti = {
            'success': [],
            'objective_value': [],
        }
        # prepare dictionary from which we create dataframe
        for free_parameter in self.free_parameters:
            pid = free_parameter.pid
            for i in free_parameter.sim_idx:
                df_dicti[f"{pid}_{i}_init"] = []
                df_dicti[f"{pid}_{i}_optimal"] = []

        # fill dictionary with initial, optimal parameters and the objective value
        for result in optimization_results:
            p_init, result_object = result
            p_opt = result_object.x
            df_dicti['success'].append(result_object.success)
            df_dicti['objective_value'].append(result_object.fun)

            # add initial and optimal parameter values
            k = 0
            for free_parameter in self.free_parameters:
                m = 0
                pid = free_parameter.pid
                for i in free_parameter.sim_idx:
                    df_dicti[f"{pid}_{i}_init"].append(p_init[k])
                    df_dicti[f"{pid}_{i}_optimal"].append(p_opt[k])
                    if free_parameter.fixed:
                        m += 1
                    else:
                        k += 1
                k += m
        df_optimization = pd.DataFrame(df_dicti)
        df_optimization.to_csv(savepath, sep='\t')


    def __compute_nsim(self):
        """Computes the number of simulations that will be done from the passed parameters"""
        sim_free = [p.sim_idx for p in self.free_parameters]
        sim_fixed = [p.sim_idx for p in self.fixed_parameters]
        sim = sim_free + sim_fixed
        sim = list(itertools.chain.from_iterable(sim))
        sim_idx = np.sort(np.unique(sim))
        # check whether simulation indices are consecutive and 0-based
        i = 0
        for idx in sim_idx:
            if idx != i:
                raise ValueError(f"Simulation indices (sim_idx) in the provided fixed/free parameters are either not "
                                 f"consecutive or 0-based: {sim_idx}")
            i += 1
        return len(np.unique(sim_idx))

    def __set_obj_fun(self, obj_fun: str):
        if obj_fun == 'default':
            self.fun = self.__obj_fun
        elif obj_fun == 'extended':
            self.fun = self.__obj_fun_ext
        else:
            raise ValueError(f"Unknown keyword ('{obj_fun}'. Use either 'default' or 'extended'.")

    ### INPUT VALIDATION (mainly parameters) ###
    def __check_parameters(self):
        self.__check_parameter_names()
        self.__check_sufficient_parameters()

    def __check_parameter_names(self):
        """Checks whether the ids of the provided parameters are correct (in the model)"""
        model_p_names = self.sim.model.fids + self.sim.model.pids
        unknown = []
        for parameters in [self.free_parameters, self.fixed_parameters]:
            for p in parameters:
                if p.pid not in model_p_names:
                    unknown.append(p.pid)
        if len(unknown) > 0:
            raise ValueError(f"Some parameter IDs are not in the model: {np.unique(unknown)}")

    def __check_sufficient_parameters(self):
        """Checks whether for each simulation all necessary parameters and initial values were provided"""
        parameters_per_sim = [[] for _ in range(self.n_sim)]
        for parameters in [self.free_parameters, self.fixed_parameters]:
            for p in parameters:
                for i in p.sim_idx:
                    parameters_per_sim[i].append(p.pid)

        missing = []
        for ids in [self.sim.model.fids, self.sim.model.pids]:
            for pid in ids:
                if not(all(pid in p for p in parameters_per_sim)):
                    missing.append(pid)
        if len(missing) > 0:
            raise ValueError(f"Insufficient parameters per simulation were provided: PROVIDED - {parameters_per_sim}; MISSING - {missing}")

    ### DEPRECATED FUNCTIONS ###
    def __obj_fun_ext_depr(self, parameters, t_end, t_step):
        """ Objective function that uses least squares """
        y0, p = self.__map_parameters(parameters)
        y_hat = self.sim.simulate(t_end, t_step, p, y0).y[[0, 2]].sum(axis=0)
        y_diff = (y_hat - self.y_ref) ** 2
        return y_diff.sum(axis=0)

    def __sample_parameter(self, lb, ub):
        if lb == 0:
            lb = 1e-10
        if ub == 0:
            ub = -1e-10
        sign_lb = np.sign(lb)
        sign_ub = np.sign(ub)
        if sign_lb * sign_ub == -1:     # lb is negative, ub is positive
            v1 = loguniform.rvs(1e-10, abs(lb)) * -1
            v2 = loguniform.rvs(1e-10, abs(ub))
            p = [(10 - abs(np.log10(abs(lb)))) / abs(10 - abs(np.log10(abs(lb))) + 10 - abs(np.log10(ub))),
                 (10 - abs(np.log10(ub))) / abs(10 - abs(np.log10(abs(lb))) + 10 - abs(np.log10(ub)))]
            v = np.random.choice([v1, v2], p=p)
        elif sign_lb == -1:   # lb and ub are negative
            v = loguniform.rvs(abs(ub), abs(lb)) * -1
        else:   # lb and ub are positive
            v = loguniform.rvs(lb, ub)
        return v
