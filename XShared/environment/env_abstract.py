from abc import ABC, abstractmethod
from XShared.environment.game import *

class Environment(ABC):
    def __init__(self):

        # DISPLAY MODE SELECTION
        # --------------------------------
        while True:
            i = input("Start environment in display mode? 'y' or 'n': ").strip().lower()
            match i:
                case 'y':
                    self._show = True
                    break
                case 'n':
                    self._show = False
                    break
                case _:
                    print("Invalid input!")
        # --------------------------------

        # ENVIRONMENT SETUP
        # --------------------------------
        self.game = SuperMarion(mode="AGENT")
        # --------------------------------
        
        # ENVIRONMENT VARIABLES
        # --------------------------------
        self._no_progress : int = 0.
        self._max_distance : float = 0.
        self._max_score : float = 0.
        self._prev_pos : tuple = (0, 0)

        self.act_space_size : int
        self.obs_space_size : int
        # --------------------------------
    
    @abstractmethod
    def _observation(self) -> tuple:
        pass
    
    @abstractmethod
    def _evaluation(self) -> float:
        pass

    def reset(self) -> tuple:
        self._no_progress = 0
        self._max_distance = 0.
        self._max_score = 0.
        self._prev_pos = (0, 0)
    
        del self.game
        self.game = SuperMarion(mode="AGENT")

        self._display()

        obs = self._observation()
        info = self._informations()

        return obs, info

    def step(self, action : int) -> tuple:

        self._action(action) # Execute action

        obs = self._observation() # Get observables

        info = self._informations() # Get metrics

        reward = self._evaluation() # Get reward

        terminated = self._termination() # Check game state
        
        return obs, reward, terminated, info

    def render(self) -> None:
        if self._show:
            self._draw()
            self._update()

    def _force_kill(self) -> None:
        for a in self.game._arena.actors():
            if isinstance(a, Player):
                self.game._arena.kill(a)

    def _action(self, action : int) -> None:
        self.game._arena.control(action)
        self.game._arena.tick(g2d.current_keys())
  
    def _termination(self) -> bool:
        return not self.game._arena.game_running()
    
    def _informations(self) -> dict:
        infos = self.game._arena.metrics()
        infos.update({'state' : self.game._arena.final_state()})
        return infos
    
    def _display(self) -> None:
        g2d.init_canvas((ARENA_W, ARENA_H), self._show)

    def _draw(self) -> None:
        g2d.clear_canvas()
        self.game._arena.draw()
    
    def _update(self) -> None:
        g2d.update_canvas()

    def _current_frame(self) -> None:
        return g2d.current_screen()