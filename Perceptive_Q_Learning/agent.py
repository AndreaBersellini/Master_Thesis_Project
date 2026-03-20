import time
from tqdm import tqdm
from scipy import stats
from datetime import datetime
from XShared.environment.env_abstract import *
from XShared.utils.organizer import *
from XShared.utils.qtable import *
from XShared.utils.agent_abstract import *

class AgentQL(Agent):
    def __init__(self, iterations : dict, epsilon_min : float, epsilon_max : float, hyperparameters : dict, paths : dict, obs_space : int, act_space : int):
        
        super().__init__(iterations, epsilon_min, epsilon_max, paths, obs_space, act_space)

        # HYPERPARAMETERS
        # --------------------------------
        self._learning_rate = hyperparameters['learning_rate']
        self._discount = hyperparameters['discount_factor']
        self._load_params()
        # --------------------------------

        # Q-TABLE
        # --------------------------------
        self._q_table = QTable(paths['table'])
        self._q_table.load()
        # --------------------------------

    def train(self, env : 'Environment') -> dict:

        previous_upperbound = 0
        self._init_result_path(datetime.now().strftime("%d-%m-%Y %H:%M"))
        start = time.time()

        for generation in range(self._generations):
            max_distances = [] # Maximum distances achieved in every episode of the current generation
            max_rewards = [] # Maximum rewards obtained in every episode of the current generation
            deltas : list = [] # Variation in the q-values
            print("\n")

            for episode in tqdm(range(self._episodes), "Gathering states"):
                episode_reward, episode_distance = 0, 0
                self._iterations += 1

                state, infos = env.reset()

                while True:
                    # DECIDE IF TAKE A RANDOM ACTION OR FOLLOW THE Q-TABLE
                    # --------------------------------
                    self._q_table()[state] = [0]*env.act_space_size if state not in self._q_table() else self._q_table()[state] # Initialize q-values for new state-action
                    
                    q_values : list = self._q_table()[state] # Q-values for each possible action
                    best_action = q_values.index(max(q_values)) # Best possible action (if values are the same return the first occurence)

                    action = self._select_action(best_action, episode_distance, 200)
                    # --------------------------------

                    # EXECUTE A STEP AND RETRIEVE THE RETURN VALUES
                    # --------------------------------
                    next_state, reward, terminated, infos = env.step(action)
                    # --------------------------------

                    # RETRIVE THE CURRENT AND THE BEST NEXT Q-VALUES
                    # --------------------------------
                    q_value = q_values[action] # Q-value corresponding to the selected action
                    self._q_table()[next_state] = [0]*env.act_space_size if next_state not in self._q_table() else self._q_table()[next_state] # Initialize q-values for new state-action
                    best_q = max(self._q_table()[next_state]) # Best q-value of next state possible actions
                    # --------------------------------

                    # CALCULATE THE NEW Q-VALUE
                    # --------------------------------
                    q_delta = self._learning_rate * (reward + self._discount * best_q - q_value)
                    self._q_table()[state][action] += q_delta
                    # --------------------------------

                    deltas.append(abs(q_delta))

                    episode_reward += reward
                    episode_distance = infos['total_distance']

                    state = next_state

                    env.render()

                    if terminated:
                        self._decrement_epsilon()

                        max_distances.append(episode_distance)
                        max_rewards.append(episode_reward)

                        # EARLY STOPPING TRIGGER
                        # --------------------------------
                        if infos['state'] == 1 and not self._complete: # --- TRIGGER THE FIRST COMPLITION ---
                            self._complete = True
                        # --------------------------------

                        break
            
            # v v v AFTER EACH GENERATION v v v

            # EXPLORATION METRICS
            # --------------------------------
            distance_mean, reward_mean, q_table_size = stats.trim_mean(max_distances, 0.1), stats.trim_mean(max_rewards, 0.1), len(self._q_table)
            self._organizer.add_train_result({'distance' : distance_mean, 'reward' : reward_mean, 'states' : q_table_size})
            self._organizer.add_train_result({'loss' : stats.trim_mean(deltas, 0)})
            self._output_log(f"Generation {generation} - Average distance traveled : {distance_mean}")
            # --------------------------------

            # AGENT TEST
            # --------------------------------
            test = self.test(env)
            self._organizer.add_test_result(test)
            self._output_log(f"Test performances - Distance : {test['distance']} - Coins : {test['coins']}")
            # --------------------------------

            self._plot_metrics()

            # ADJUST EXPLORATION PARAMETERS 
            # --------------------------------
            previous_upperbound = self._distance_upperbound if self._distance_upperbound >= previous_upperbound else previous_upperbound
            self._epsilon_episode = self._epsilon_max
            self._distance_upperbound = test['distance']
            # --------------------------------
    
             # SAVE Q-TABLE AND HYPERPARAMETERS
            # --------------------------------
            self._q_table.save()
            self._save_params()
            # --------------------------------

            # EARLY STOPPING
            # --------------------------------
            if self._early_stopping(30): break
            # --------------------------------

        self._timer = time.time() - start
        self._end_train()

    def test(self, env : 'Environment') -> dict:
            
        total_reward = 0
        state, infos = env.reset()
        
        while True:
            self._q_table()[state] = [0]*env.act_space_size if state not in self._q_table() else self._q_table()[state]
            
            q_values : list = self._q_table()[state]

            action = q_values.index(max(q_values))

            state, reward, terminated, infos = env.step(action)

            total_reward += reward

            env.render()

            if terminated:
                metrics = {'steps' : infos['frames'],
                        'distance' : infos['total_distance'],
                        'reward' : total_reward,
                        'state' : infos['state'],
                        'coins' : infos['score']}

                return metrics