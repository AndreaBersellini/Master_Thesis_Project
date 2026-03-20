import torch
import torch.nn as nn
import torch.optim as optim

class DQNetwork(nn.Module):
    def __init__(self, stack : int, input_size : int, output_size : int, lr : float, path : str):
        super(DQNetwork, self).__init__()

        self._input_size = input_size * stack
        self._output_size = output_size
        self._path = path

        self.network = nn.Sequential(
            nn.Linear(self._input_size, 64),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(64, 32),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(32, 8),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(8, self._output_size)
        )

        #nn.init.constant_(self.network[-1].weight, 0.0)
        #nn.init.constant_(self.network[-1].bias, 0.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def save(self) -> None:
        torch.save(self.state_dict(), self._path)

    def load(self) -> bool:
        try:
            self.load_state_dict(torch.load(self._path, weights_only=True))
        except:
            pass