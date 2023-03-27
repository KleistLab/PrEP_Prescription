import numpy as np
from typing import List, Union
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
from src.models.models import Model
import copy

class SimResults:
    def __init__(self, t, y):
        self.t = t
        self.y = y

class Simulator:

    def __init__(self, model: Model):
        self.model = model

    def __integrate(self, t_end: int, t_step: float, parameters: Union[np.ndarray, List[int]], y0: Union[np.ndarray, List[int]], **kwargs):
        """ Solves the ODE-System (models) and evaluates the functions from 0 to t_end
        Returns t and the respective y-values

        :param t_end: Endpoint of the simulation
        :param t_step: Steps at which the function will be evaluated (t_eval will be np.arange(0, t_end, t_step)
        :param parameters: Model parameters
        :param y0: Initial values
        :param kwargs: Hyperparmeters used in scipy.integrate.solve_ivp()
        :return: t_eval, y(t)
        """
        t_eval = np.arange(0, t_end, t_step)
        solution = solve_ivp(fun=self.model.fun, t_span=(0, t_end), t_eval=t_eval, y0=y0, args=(parameters, ), **kwargs)
        return solution.t, solution.y

    def simulate(self, endpoints: List[int], t_step: int, parameters: List, y0: List, smooth: bool = False, **kwargs):
        """ Simulates the models by calling __integrate()
        The function can run multiple subsequent simulations with different parameters
        Simulations are run in the intervals (0, t_ends[0]), (t_ends[0]+t_step, t_ends[1]), ...

        :param endpoints: Endpoint of each simulation
        :param t_step: Steps size for function evaluation
        :param parameters: parameter sets for each simulation
        :param y0: initial values for each simulation
        :param smooth: Set to True if the simulation should be smoothed (by moving average)
        :param kwargs:
        :return:
        """
        #FIXME: Check whether t_ends, parameters, y0 all have the same length. Otherwise, throw error

        endpoints = list(endpoints) # code would fail if <endpoints> was a numpy array. Make sure it's a list
        t_start = [0] + [ep + 1 for ep in endpoints]    # simulation should always start at endpoint + delta_t
        #t_start = [0] + endpoints    # simulation should always start at endpoint + delta_t
        t_ends = np.diff(t_start)   # compute t_end for each simulation, assuming t_start=0

        # run all simulations
        t = np.array([])
        for k, t_end in enumerate(t_ends):
            p = parameters[k]
            y_init = y0[k]
            sol_t, sol_y = self.__integrate(t_end, t_step, p, y_init, **kwargs)

            # collect results
            t = np.append(t, t_start[k] + sol_t)
            if k == 0:
                y = sol_y
            else:
                y = np.hstack((y, sol_y))

        if smooth:
            y = self.__smooth_simulation(y)
        results = SimResults(t, y)
        return results

    def simulate_continuous(self, endpoints: List[int], t_step: int, parameters: List, y0: List, smooth: bool = False, **kwargs):
        """ Simulates the models by calling __integrate()
        The function can run multiple subsequent simulations with different parameters
        Simulations are run in the intervals (0, t_ends[0]), (t_ends[0]+t_step, t_ends[1]), ...
        Unlike simulate(), simulate_continuous() does not use different initial values for each simulation. Instead, the
        function uses the last y-value from the previous simulation as initial value.

        :param endpoints: Endpoint of each simulation
        :param t_step: Steps size for function evaluation
        :param parameters: parameter sets for each simulation
        :param y0: initial values for first simulation
        :param smooth: Set to True if the simulation should be smoothed (by moving average)
        :param kwargs:
        :return:
        """
        #FIXME: Check whether t_ends, parameters, y0 all have the same length. Otherwise, throw error
        t_start = [0] + endpoints
        t_ends = np.diff(t_start) + 1   # compute t_end for each simulation, assuming t_start=0
                                        # each simulation takes +1, since they start at the endpoint of the previous one
        # run all simulations
        t = np.array([])
        y_init = y0[0]
        for k, t_end in enumerate(t_ends):
            p = parameters[k]
            sol_t, sol_y = self.__integrate(t_end, t_step, p, y_init, **kwargs)

            # collect results
            #t = np.append(t, t_start[k] + sol_t)
            if k == 0:
                t = sol_t
                y = sol_y
            else:
                t = np.append(t, t_start[k] + sol_t[:-1])
                y = np.hstack((y, np.delete(sol_y, 0, axis=1)))
            y_init = np.take(y, -1, axis=1)

        if smooth:
            y = self.__smooth_simulation(y)
        results = SimResults(t, y)
        return results

    def __smooth_simulation(self, sim_results):
        window_size = 3
        nrow, ncol = sim_results.shape
        sim_results_smooth = np.zeros((nrow, ncol-2))
        for k, y in enumerate(sim_results):
            sim_results_smooth[k] = lfilter(np.ones(window_size) / window_size, 1, y)[2:]       # remove first 2 points as they are averaged with 0s
        return sim_results_smooth
