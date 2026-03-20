import time
from tqdm import tqdm
import numpy as np
from scipy import stats
from datetime import datetime
from XShared.environment.env_abstract import *
from XShared.utils.agent_abstract import *
from XShared.utils.organizer import *
from XShared.utils.dqn_linear import *
from XShared.utils.buffer import *
        
class AgentDQL(Agent):
    def __init__(self, iterations : dict, epsilon_min : float, epsilon_max : float, hyperparameters : dict, paths : dict, obs_space : int, act_space : int):

        super().__init__(iterations, epsilon_min, epsilon_max, paths, obs_space, act_space)

        # --- HYPERPARAMETERS ---

        self._learning_rate = hyperparameters['learning_rate']
        self._discount = hyperparameters['discount_factor']
        self._batch_size = hyperparameters['batch_size']
        self._buffer_size = hyperparameters['buffer_size']
        self._threshold = hyperparameters['threshold']
        self._stack = 3

        # --- REPLAY BUFFER ---

        self._replay_buffer = ReplayBuffer(self._buffer_size, self._threshold, self._paths['replay_buff'])
        #self._replay_buffer = SelectiveExperienceReplay(self._buffer_size, self._threshold, self._paths['replay_buff'])
        self._replay_buffer.load()

        # --- DEEP Q-NETWORK ---
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._loss = nn.SmoothL1Loss()#nn.MSELoss()##nn.HuberLoss()

        torch.manual_seed(69)
        self._policy_network = DQNetwork(self._stack, self._obs_space, self._act_space, self._learning_rate, self._paths['policy_net']).to(self._device)
        torch.manual_seed(69)
        self._target_network = DQNetwork(self._stack, self._obs_space, self._act_space, self._learning_rate, self._paths['target_net']).to(self._device)
        
        self._policy_network.load()
        self._target_network.load()
        self._load_params()
    
    def _train_network(self) -> tuple:

        self._policy_network.train()

        states, actions, next_states, rewards, terminals = zip(*self._replay_buffer.batch(self._batch_size))

        states = torch.stack(states)
        actions = torch.stack(actions).unsqueeze(1)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards).unsqueeze(1)
        terminals = torch.stack(terminals).unsqueeze(1)

        self._policy_network.optimizer.zero_grad()

        q_pred : torch.Tensor = self._policy_network(states)
        q_pred.retain_grad()

        q_pred = q_pred.gather(dim=1, index=actions)

        with torch.no_grad():
            target_act : torch.Tensor = torch.argmax(self._policy_network(next_states), dim=1).unsqueeze(1)
            target_pred : torch.Tensor = self._target_network(next_states)
            q_best = target_pred.gather(dim=1, index=target_act)

        q_target : torch.Tensor = rewards + self._discount * q_best * (1 - terminals)

        loss : torch.Tensor = self._loss(q_pred, q_target)
        loss.backward()
        self._policy_network.optimizer.step()

        self._policy_network.eval()

        return loss.item()

    def _combine_states(self, memory : deque):
        return np.concatenate(memory, axis=0)

    def train(self, env : 'Environment') -> dict:

        previous_upperbound = 0
        self._init_result_path(datetime.now().strftime("%d-%m-%Y %H:%M"))
        start = time.time()

        for generation in range(self._generations):
            max_distances : list = [] # Maximum distances achieved during each episode of the current generation
            max_rewards : list = [] # Maximum rewards obtained during each episode of the current generation
            training_losses : list = []
            print("\n")

            for episode in tqdm(range(self._episodes), "Gathering states"):
                episode_reward, episode_distance = 0, 0
                self._iterations += 1

                state, infos = env.reset()

                state_memory = deque([state] * self._stack, maxlen=self._stack)
                state = self._combine_states(state_memory)
                
                while True:
                    with torch.no_grad():
                        q_values = self._policy_network(torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0))

                    best_action = torch.argmax(q_values).item()
                        
                    #print(f"State: {state}")
                    #print(f"Qs: {q_values}, Best:{best_action}\n")
                    #time.sleep(0.5)
                    
                    action = self._select_action(best_action, episode_distance, 200)

                    next_state, reward, terminated, infos = env.step(action)
                    #print(f"Action:{action}, Reward:{reward}\n")

                    state_memory.append(next_state)
                    next_state = self._combine_states(state_memory)

                    self._replay_buffer.push((state, action, next_state, reward, terminated))

                    episode_reward += reward
                    episode_distance = infos['total_distance']

                    state = next_state

                    env.render()

                    # v v v AFTER EACH STEP v v v

                    # DQN TRAIN
                    # --------------------------------
                    if self._replay_buffer.ready():
                        batch_loss = self._train_network()
                        training_losses.append(batch_loss)
                    # --------------------------------

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
                
                # v v v AFTER EACH EPISODE v v v
                
            # TARGET NETWORK UPDATE
            # --------------------------------
            if generation % 20 == 0 : self._target_network.load_state_dict(self._policy_network.state_dict())
            # --------------------------------

            # v v v AFTER EACH GENERATION v v v

            # EXPLORATION METRICS
            # --------------------------------
            distance_mean, reward_mean, stored_samples = stats.trim_mean(max_distances, 0.1), stats.trim_mean(max_rewards, 0.1), len(self._replay_buffer)
            self._organizer.add_train_result({'distance' : distance_mean, 'reward' : reward_mean, 'states' : stored_samples})
            self._organizer.add_train_result({'loss' : stats.trim_mean(training_losses, 0)})
            self._output_log(f"Generation {generation} - Average distance traveled : {distance_mean} - Replay buffer samples : {stored_samples}")
            # --------------------------------

            #self._replay_buffer.display_state(True, True)

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

            # SAVE MODEL
            # --------------------------------
            self._policy_network.save()
            self._target_network.save()
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

        state_memory = deque([state] * self._stack, maxlen=self._stack)
        state = self._combine_states(state_memory)
        
        while True:
            with torch.no_grad():
                q_values = self._policy_network(torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0))

            action = torch.argmax(q_values).item()

            state, reward, terminated, infos = env.step(action)
            state_memory.append(state)
            state = self._combine_states(state_memory)
            total_reward += reward
            env.render()

            if terminated:
                metrics = {'steps' : infos['frames'],
                        'distance' : infos['total_distance'],
                        'reward' : total_reward,
                        'state' : infos['state'],
                        'coins' : infos['score']}

                return metrics