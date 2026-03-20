import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def plot_feature_maps(feature_maps : list[torch.Tensor], titles : list[str], path : str) -> None:

    for l_idx, (title, f) in enumerate(zip(titles, feature_maps)):
        f = f.squeeze(0) # [C: number of channels, H: image heigh, W: image width]
        C, H, W = f.shape[0], f.shape[1], f.shape[2]

        cols, rows = 8, math.ceil(C / 8)
        plt.figure(figsize=(cols, rows))
        plt.suptitle(f"{title} Feature Maps")

        for i in range(C):
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(f[i].detach().cpu().numpy(), cmap='viridis')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{path}/fm_{l_idx}_{title}.pdf", bbox_inches='tight', format='pdf')
        plt.close()

class localConv2d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, patch_size : int, kernel : int, stride : int):
        super().__init__()
        self._patch_size = patch_size
        self._out_channels = out_channels
        self._conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
    
    def forward(self, x : torch.Tensor):
        batch_size = x.shape[0]
        x = x.unfold(2, self._patch_size, self._patch_size).unfold(3, self._patch_size, self._patch_size)
        x = x.contiguous().view(x.shape[0] * x.shape[2] * x.shape[3], x.shape[1], self._patch_size, self._patch_size)
        x = self._conv(x)
        x = x.view(batch_size, self._out_channels * (x.shape[0] // batch_size), x.shape[2], x.shape[3])
        return x

class PatchLayer(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, patch_size : int):
        super().__init__()
        self._patch_size = patch_size
        self._out_channels = out_channels
        #self._conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
    
    def forward(self, x : torch.Tensor):
        x = x.unfold(2, self._patch_size, self._patch_size).unfold(3, self._patch_size, self._patch_size)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4], x.shape[5])
        return x
    
class ImageDQN(nn.Module):
    def __init__(self, input_channels, input_size, output_size, lr : float, model_path : str, plot_path : str):
        super(ImageDQN, self).__init__()

        self._input_channels = input_channels
        self._input_size = input_size
        self._output_size = output_size
        self._model_path = model_path
        self._plot_path = plot_path

        self._grid_size = 5
        self._features_x_patch = 4 # number of different features extracted from each patch
        self._kernel_size = 3 #(self._input_size // 2) # size of the kernels of the first layer
        self._patch_size = self._input_size // self._grid_size
        self._feature_channels = self._grid_size**2 * self._features_x_patch
        self._patch_layer_features = self._input_channels * self._grid_size**2

        self._network = nn.Sequential(
            #localConv2d(self._input_channels, self._features_x_patch, self._patch_size, self._kernel_size, 1),
            PatchLayer(self._input_channels, self._patch_layer_features, self._patch_size),
            #nn.InstanceNorm2d(self._patch_layer_features),
            nn.BatchNorm2d(self._patch_layer_features),
            nn.ReLU(),
            nn.Conv2d(self._patch_layer_features, 32, 5, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, self._output_size)
        )

        nn.init.constant_(self._network[-1].weight, 0.0)
        nn.init.constant_(self._network[-1].bias, 0.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self._network):
            #print(f"layer {i}:\n", x.size)
            x = layer(x)
            #print(x.shape, "\n-------")
        return x
    
    def features(self, x : torch.Tensor) -> None:
        f_maps : list = []
        l_cat : list = []

        for i, layer in enumerate(self._network):
            x = layer(x)
            if isinstance(layer, nn.modules.conv._ConvNd) or isinstance(layer, PatchLayer):
                f_maps.append(x)
                l_cat.append(layer.__class__.__name__)

        plot_feature_maps(f_maps, l_cat, self._plot_path)
    
    def save(self) -> None:
        torch.save(self.state_dict(), self._model_path)

    """def load(self) -> bool:
        try:
            self.load_state_dict(torch.load(self._model_path, weights_only=True))
        except:
            pass"""
    def load(self) -> bool:
        try:
            state_dict = torch.load(self._model_path)
            self.load_state_dict(state_dict)
            self.eval()
            return True
        except Exception as e:
            print(f"Loading failed: {e}")
            return False