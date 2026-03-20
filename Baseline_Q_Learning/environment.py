import math
from XShared.environment.game.super_marion import *
from XShared.environment.env_abstract import Environment

class CustomEnv(Environment):
    def __init__(self):
        super().__init__()

        self.act_space_size = 4
        self.obs_space_size = 2

        self._rewards = {
            'death' : -1000,
            'victory' : 1000,
            'progress' : 1,
            'coin' : 10,
            'stall' : -1
        }

    def _evaluation(self):

        match self.game._arena.game_running():
            case True:

                # GAME VARIABLES
                # --------------------------------
                metrics = self.game._arena.metrics()
                for a in self.game._arena.actors():
                    if isinstance(a, Player):
                        px, (_, py), (_, sy) = a.absolute_position(), a.position(), a.size()
                # --------------------------------

                # TOTAL PROGRESS OF THE STEP
                # --------------------------------
                delta_dist = metrics['total_distance'] - self._max_distance
                delta_score = metrics['score'] - self._max_score
                stalling = True if (px, py) == self._prev_pos else False

                self._max_distance = metrics['total_distance']
                self._max_score = metrics['score']
                self._prev_pos = (px, py)
                # --------------------------------

                # CHECK FOR NO PROGRESS
                # --------------------------------
                self._no_progress = self._no_progress + 1 if delta_dist == 0 else 0
                if self._no_progress >= 100:
                    self._force_kill()
                    return self._rewards['stall']
                # --------------------------------
                
                # CHECK FOR STATIONARY
                # --------------------------------
                if stalling : return self._rewards['stall']
                # --------------------------------

                # CHECK FOR ACTOR FALLING IN A HOLE
                # --------------------------------
                grounded = False
                for other in self.game._arena.collisions():
                    if isinstance(other, Ground):
                        grounded = True

                if not grounded and (py + sy - 10) > (ARENA_H - GROUND_H):
                    self._force_kill()
                    return self._rewards['death']
                # --------------------------------
                
                # REWARD FOR PROGRESS AND COINS
                # --------------------------------
                return  ((delta_dist * self._rewards['progress']) + 
                        (delta_score * self._rewards['coin']))
                # --------------------------------
            

            case False:
                match self.game._arena.final_state():
                    case 1:

                        # VICTORY REWARD
                        # --------------------------------
                        return self._rewards['victory']
                        # --------------------------------

                    case 2:

                        # DEFEAT REWARD
                        # --------------------------------
                        return self._rewards['death']
                        # --------------------------------

                    case _:
                        raise f"GAME IS NOT RUNNING BUT FINAL STATE IS UNCLEAR -> {str(self.game._arena.final_state())}"
    
    def _observation(self):

        sampling = 1 # Sampling unit (1 = no sampling)

        px, py = 0, 0

        # PLAYER POSITION
        # --------------------------------
        for a in self.game._arena.actors():
            if isinstance(a, Player):
                px = a.absolute_position()
                _, py = a.position()

        px = sampling * math.floor(px / sampling)
        py = sampling * math.floor(py / sampling)
        # --------------------------------

        return tuple([int(px), int(py)])