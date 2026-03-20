import os
import numpy as np
import random
from abc import ABC, abstractmethod
from XShared.environment.env_abstract import Environment
from XShared.utils.organizer import Organizer

class Agent(ABC):

    def __init__(self, iterations : dict, epsilon_min : float, epsilon_max : float, paths : dict, obs_space : int, act_space : int) -> None:
        
        # NUMBER OF ITERATIONS
        # --------------------------------
        self._generations : int = iterations['generations']
        self._episodes : int = iterations['episodes']
        # --------------------------------

        # EXPLORATION PARAMETERS
        # --------------------------------
        self._epsilon_min : float = epsilon_min
        self._epsilon_max : float = epsilon_max
        self._epsilon_decay : float = 1 - self._epsilon_max / self._episodes
        self._epsilon_episode : float = self._epsilon_max
        self._distance_upperbound : float = 0
        # --------------------------------

        # ENVIRONMENT DIMENSIONS
        # --------------------------------
        self._act_space : int = act_space
        self._obs_space : int = obs_space
        # --------------------------------

        # AGENT ARCHIVE PATHS
        # --------------------------------
        self._paths : dict = paths
        # --------------------------------

        # TRAINING METRICS
        # --------------------------------
        self._timer : float = 0
        self._iterations : int = 0
        self._unique_states : int = 0
        self._complete : bool = False
        # --------------------------------

        # RESULT ORGANIZER
        # --------------------------------
        self._organizer = Organizer(self._paths['results'])
        # --------------------------------

        # SUPPORT VARIABLES
        # --------------------------------
        self._conv_count : int = 0
        # --------------------------------

    def _load_params(self) -> None:
        if os.path.exists(self._paths['params']):
            params = np.load(self._paths['params'])
            self._distance_upperbound = params[0]
    
    def _save_params(self) -> None:
        np.save(self._paths['params'], np.array([self._distance_upperbound]))

    def _compute_epsilon(self, curr_dist : float, intervall : float) -> float:
        starting_point = self._distance_upperbound - intervall

        match curr_dist:
            case _ if curr_dist <= starting_point:
                epsilon = self._epsilon_min
            case _ if curr_dist >= self._distance_upperbound:
                epsilon = self._epsilon_episode
            case _:
                mx = self._distance_upperbound - starting_point
                x = curr_dist - starting_point
                epsilon = self._epsilon_min + (self._epsilon_episode - self._epsilon_min) * ((x / mx))
        
        return epsilon

    def _select_action(self, best_action : int, curr_dist : float, intervall : float) -> float:
        
        epsilon = self._compute_epsilon(curr_dist, intervall)

        if random.uniform(0, 1) < epsilon:
            return np.random.choice(range(self._act_space))
        else:
            return best_action
            
    def _decrement_epsilon(self) -> None:
        if self._epsilon_episode >= self._epsilon_min:
            self._epsilon_episode *= self._epsilon_decay
             
    def _early_stopping(self, tollerance : float) -> bool:
        stop = False
        if self._complete:
            if (4230 - tollerance) <= self._distance_upperbound and  self._distance_upperbound <= (4230 + tollerance):
                self._conv_count += 1
            else:
                self._conv_count = 0
            
            if self._conv_count == 5 : stop = True

        return stop
        
    def _init_result_path(self, timestamp : str) -> None:
        if not os.path.exists(self._paths['results']):
            os.makedirs(self._paths['results'])

        with open(os.path.join(self._paths['results'], "__logs.txt"), "a") as file:
            file.write("\n" + "-"*20 + f"Training start: {timestamp}" + "-"*20 + "\n\n")

    def _plot_metrics(self) -> None:
            self._organizer.plot_train_metrics()
            self._organizer.plot_test_metrics()
            self._organizer.plot_disc_states()
            self._organizer.plot_losses()

    def _output_log(self, message : str) -> None:
        with open(os.path.join(self._paths['results'], "__logs.txt"), "a") as file:
            file.write(message + "\n")
            print(message)

    def _end_train(self) -> None:
        self._organizer.export_results()
        end_log = f"Train finished!\nTotal iterations: {self._iterations} - Training time: {self._timer}"
        self._output_log(end_log)

    @abstractmethod
    def train(self, env : 'Environment') -> None:
        pass
    
    @abstractmethod
    def test(self, env : 'Environment') -> dict:
        pass