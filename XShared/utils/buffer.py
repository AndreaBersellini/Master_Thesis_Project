import cv2
import random
import torch
import numpy as np
import pickle as pkl
from collections import deque
from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self, size : int, threshold : int, path : str):
        self._size = size
        self._threshold = threshold
        self._path = path

        self._buffer = deque(maxlen=size)

    def push(self, transition : tuple) -> None:

        state, action, next_state, reward, terminal = transition

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)

        self._buffer.append((state, action, next_state, reward, terminal))

    def batch(self, size : int) -> list:
        return random.sample(self._buffer, size)
    
    def clear(self) -> None:
        self._buffer.clear()
    
    def ready(self) -> bool:
        if self._threshold >= len(self._buffer) : return False
        return True

    def current_size(self) -> None:
        print(f"Buffer size : {len(self._buffer)}")

    def display_state(self, terminal : bool):

        filter_buffer = [(s, a, n, r, t) for (s, a, n, r, t) in self._buffer if t == float(terminal)]
        states = random.sample(filter_buffer, 1)
        
        for i, (state, _, next_state, _, _) in enumerate(states):
            state, next_state = state.numpy(), next_state.numpy()
            state, next_state = np.hstack(state), np.hstack(next_state)

            cv2.imshow(f"State {i}", state)
            cv2.imshow(f"Next State {i}", next_state)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()

    def save(self) -> None:
        with open(self._path + '.pkl', 'wb') as file:
            pkl.dump(self._buffer, file)

    def load(self) -> None:
        try:
            with open(self._path + '.pkl', 'rb') as file:
                self._buffer = pkl.load(file)
        except:
            pass
    
    def __len__(self):
        return len(self._buffer)
    
class ActionSelectiveReplayBuffer(Dataset):
    def __init__(self, action_space : int, size : int, threshold : int, path : str):
        self._act_space = action_space
        self._size = size
        self._threshold = threshold
        self._path = path

        # --- COLLECTION OF ALL KNOWN STATES ---

        self._known_states : list[set] = []
        self._buffers : list[deque] = []

        for _ in range(self._act_space):
            self._buffers.append(deque(maxlen=(self._size // self._act_space)))
            self._known_states.append(set())

    def push(self, state : tuple, action : int, next_state : tuple, reward : float) -> None:
        #if (state, next_state, reward) not in self._known_states[action]:
            #self._known_states[action].add((state, next_state, reward))

            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            self._buffers[action].append((state, next_state, reward))

    def batch(self, size : int, action : int) -> list:
        return random.sample(self._buffers[action], size)
    
    def ready(self) -> bool:
        for buffer in self._buffers:
            if self._threshold >= len(buffer) : return False
        return True

    def current_size(self) -> None:
        for i, buffer in enumerate(self._buffers):
            print(f"Buffer [action:{i}] size : {len(buffer)}")

    def clear(self) -> None:
        for buffer in self._known_states:
            buffer.clear()
        for buffer in self._buffers:
            buffer.clear()
    """
    def save(self) -> None:
        with open(self._path + '.pkl', 'wb') as file:
            pkl.dump(self._buffer, file)

    def load(self) -> None:
        try:
            with open(self._path + '.pkl', 'rb') as file:
                self._buffer = pkl.load(file)
        except:
            pass
    """

    def __len__(self):
        return sum(len(buff) for buff in self._buffers)

class SelectiveExperienceReplay(Dataset):
    def __init__(self, size : int, threshold : int, path : str):
        self._size = size
        self._threshold = threshold
        self._path = path

        # --- COLLECTION OF ALL KNOWN STATES ---

        #self._known_states = set()

        # --- STORES EXPERIENCES BASED ON THE REWARD TYPE ---

        self._positive_buffer = deque(maxlen=size)
        self._negative_buffer = deque(maxlen=size)
        self._zero_buffer = deque(maxlen=size)

    def push(self, transition : tuple) -> None:

        #if transition not in self._known_states:

            #self._known_states.add(transition)

        state, action, next_state, reward, terminal = transition

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)

        match reward:
            case _ if reward > 0:
                self._positive_buffer.append((state, action, next_state, reward, terminal))
            case _ if reward < 0:
                self._negative_buffer.append((state, action, next_state, reward, terminal))
            case _ if reward == 0:
                self._zero_buffer.append((state, action, next_state, reward, terminal))
    
    def ready(self) -> bool:
        return all(self._threshold < length for length in[len(self._positive_buffer), len(self._negative_buffer), len(self._zero_buffer)])

    def current_size(self) -> None:
        print(f"P_buffer: {len(self._positive_buffer)}, N_buffer: {len(self._negative_buffer)}, Z_buffer: {len(self._zero_buffer)}")

    def batch(self, size : int) -> list:
        batch = random.sample(self._positive_buffer, size//3) + random.sample(self._negative_buffer, size//3) + random.sample(self._zero_buffer, size//3)
        random.shuffle(batch)
        return batch

    def clear(self) -> None:
        #self._known_states.clear()
        self._positive_buffer.clear()
        self._negative_buffer.clear()
        self._zero_buffer.clear()

    def save(self) -> None:
        with open(self._path + '.pkl', 'wb') as file:
            pkl.dump((self._positive_buffer, self._negative_buffer, self._zero_buffer), file)

    def load(self) -> None:
        try:
            with open(self._path + '.pkl', 'rb') as file:
                self._positive_buffer, self._negative_buffer, self._zero_buffer = pkl.load(file)
        except:
            pass
    
    def __len__(self):
        return min(len(self._positive_buffer), len(self._negative_buffer), len(self._zero_buffer))

    
class PrioritizedExperienceReplay():
    def __init__(self, size : int, threshold : int, path : str):
        self._threshold = threshold
        self._path = path

        self._buffer = deque(maxlen=size)

    def push(self, transition : tuple) -> None:

        state, action, next_state, reward, terminal = transition

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)

        self._buffer.append((state, action, next_state, reward, terminal))

    def batch(self, size : int) -> list:
        return random.sample(self._buffer, size)
    
    def ready(self) -> bool:
        if self._threshold >= len(self._buffer) : return False
        return True

    def display_state(self, terminal : bool):
        filter_buffer = [(s, a, n, r, t) for (s, a, n, r, t) in self._buffer if t == float(terminal)]
        states = random.sample(filter_buffer, 1)
        
        for i, (state, _, next_state, _, _) in enumerate(states):
            state, next_state = state.numpy(), next_state.numpy()
            state, next_state = np.hstack(state), np.hstack(next_state)

            cv2.imshow(f"State {i}", state)
            cv2.imshow(f"Next State {i}", next_state)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()

    def save(self) -> None:
        with open(self._path + '.pkl', 'wb') as file:
            pkl.dump(self._buffer, file)

    def load(self) -> None:
        try:
            with open(self._path + '.pkl', 'rb') as file:
                self._buffer = pkl.load(file)
        except:
            pass
    
    def __len__(self):
        return len(self._buffer)