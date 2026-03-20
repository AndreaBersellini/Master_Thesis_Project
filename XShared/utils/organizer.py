import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Organizer():
    def __init__(self, path : str):

        # RESULTS ARCHIVE PATH
        # --------------------------------
        self._path : str = path
        # --------------------------------

        # TRAINING METRICS
        # --------------------------------
        self._avg_distances : list = []
        self._avg_rewards : list = []
        self._new_states : list = []
        self._epi_actions : list = []
        # --------------------------------

        # TEST PERFORMANCES
        # --------------------------------
        self._test_distances : list = []
        self._test_rewards : list = []
        # --------------------------------

        # NN METRICS
        # --------------------------------
        self._losses : list = []
        # --------------------------------

    def export_results(self) -> None:
        with open(f'{self._path}/_dst.pkl', 'wb') as file:
            pkl.dump(self._avg_distances, file)

        with open(f'{self._path}/_rwd.pkl', 'wb') as file:
            pkl.dump(self._avg_rewards, file)

        with open(f'{self._path}/_sts.pkl', 'wb') as file:
            pkl.dump(self._new_states, file)

    def add_train_result(self, result : dict) -> None:
        if 'distance' in result : self._avg_distances.append(result['distance'])
        if 'reward' in result : self._avg_rewards.append(result['reward'])
        if 'states' in result : self._new_states.append(result['states'])
        if 'explorations' in result and 'exploitations' in result : self._epi_actions.append((result['explorations'], result['exploitations']))
        if 'loss' in result : self._losses.append(result['loss'])

    def add_test_result(self, result : dict) -> None:
        self._test_distances.append(result['distance'])
        self._test_rewards.append(result['reward'])

    def plot_test_metrics(self) -> None:

        figure, (axis1, axis2) = plt.subplots(2, figsize=(14, 8))

        # DISTANCE PLOT
        # --------------------------------
        axis1.plot(list(range(len(self._test_distances))), self._test_distances, color='blue')
        axis1.set_title("Agent Performance [Distance]")
        axis1.set_ylabel("Distance")
        axis1.grid()
        axis1.xaxis.set_major_locator(MaxNLocator(integer=True))
        # --------------------------------

        # REWARD PLOT
        # --------------------------------
        axis2.plot(list(range(len(self._test_rewards))), self._test_rewards, color='darkorange')
        axis2.set_title("Agent Performance [Reward]")
        axis2.set_ylabel("Reward")
        axis2.grid()
        axis2.xaxis.set_major_locator(MaxNLocator(integer=True))
        # --------------------------------

        plt.subplots_adjust(hspace=0.4)
        
        plt.savefig(f"{self._path}/test_plot.pdf", bbox_inches='tight', format='pdf')
        plt.close(figure)

    def plot_train_metrics(self) -> None:
        
        figure, (axis1, axis2) = plt.subplots(2, figsize=(14, 8))

        # DISTANCE PLOT
        # --------------------------------
        axis1.plot(list(range(len(self._avg_distances))), self._avg_distances, color='blue')
        axis1.set_title("Training Performances [Generation Average Distance]")
        axis1.set_ylabel("Average distance")
        axis1.grid()
        axis1.xaxis.set_major_locator(MaxNLocator(integer=True))
        # --------------------------------

        # REWARD PLOT
        # --------------------------------
        axis2.plot(list(range(len(self._avg_rewards))), self._avg_rewards, color='darkorange')
        axis2.set_title("Training Performances [Generation Average Reward]")
        axis2.set_ylabel("Average reward")
        axis2.grid()
        axis2.xaxis.set_major_locator(MaxNLocator(integer=True))
        # --------------------------------

        plt.subplots_adjust(hspace=0.4)

        plt.savefig(f"{self._path}/train_plot.pdf", bbox_inches='tight', format='pdf')
        plt.close(figure)

    def plot_expl_comp(self) -> None:
        exploration, exploitation = zip(*self._epi_actions)
        x = range(len(self._epi_actions))
        y = np.vstack([exploration, exploitation])

        labels = ["Avg. Explorations", "Avg. Exploitations"]

        plt.stackplot(x, y, labels=labels)
        plt.title("Explorations vs Exploitations")
        plt.grid()
        plt.legend(loc='upper left')
        plt.savefig(f"{self._path}/expl_plot.pdf", bbox_inches='tight', format='pdf')
        plt.close()

    def plot_disc_states(self) -> None:
        plt.plot(list(range(len(self._new_states))), self._new_states, color='blue')
        plt.title("New states discovery")
        plt.ylabel("States")
        plt.grid()

        plt.savefig(f"{self._path}/states_plot.pdf", bbox_inches='tight', format='pdf')
        plt.close()

    def plot_losses(self) -> None:
        plt.plot(list(range(len(self._losses))), self._losses, color='blue')
        plt.title("Training loss")
        plt.ylabel("Loss")
        plt.grid()

        plt.savefig(f"{self._path}/loss_plot.pdf", bbox_inches='tight', format='pdf')
        plt.close()