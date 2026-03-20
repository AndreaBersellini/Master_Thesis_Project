import math
from XShared.environment.game.super_marion import *
from XShared.environment.env_abstract import Environment

class CustomEnv(Environment):
    def __init__(self):
        super().__init__()

        self.act_space_size = 4
        self.obs_space_size = 8

        self._range = 500

        # ---- EVALUATION VARIABLES ----

        self._rewards = {
            'death' : -1,
            'victory' : 100,
            'progress' : 0.0001,
            'coin' : 100,
            'stall' : -0.01
        }

    # ---- ENVIRONMENT INTERFACE FUNCTIONS ----

    def _evaluation(self) -> float:
        match self.game._arena.game_running():
            case True:
                # --- GAME VARIABLES ---
                metrics = self.game._arena.metrics()
                for a in self.game._arena.actors():
                    if isinstance(a, Player):
                        px, (_, py), (_, sy) = a.absolute_position(), a.position(), a.size()

                 # --- CALCULATE THE TOTAL PROGRESS OF THE STEP ---

                delta_dist = metrics['total_distance'] - self._max_distance
                delta_score = metrics['score'] - self._max_score
                stalling = True if (px, py) == self._prev_pos else False

                self._max_distance = metrics['total_distance']
                self._max_score = metrics['score']
                self._prev_pos = (px, py)

                # --- CHECK IF NO PROGRESS IS MADE ---

                self._no_progress = self._no_progress + 1 if delta_dist == 0 else 0
                if self._no_progress >= 100:#500:
                    self._force_kill()
                    return self._rewards['stall']
                
                # --- CHECK IF STATIONARY ---

                if stalling : return self._rewards['stall']

                # --- CHECK IF THE ACTOR IS FALLING IN A HOLE ---
                grounded = False
                for other in self.game._arena.collisions():
                    if isinstance(other, Ground):
                        grounded = True

                if not grounded and (py + sy - 10) > (ARENA_H - GROUND_H):
                    self._force_kill()
                    return self._rewards['death']
                
                # --- REWARD FOR PROGRESS AND COINS ---

                return  ((delta_dist * self._rewards['progress']) + 
                        (delta_score * self._rewards['coin']))
            
            case False:
                match self.game._arena.final_state(): # --- FINAL GAME STATE ---
                    case 1:
                        return self._rewards['victory'] # --- VICTORY REWARD ---
                    case 2:
                        return self._rewards['death'] # --- DEFEAT REWARD ---
                    case _:
                        raise f"GAME IS NOT RUNNING BUT FINAL STATE IS UNCLEAR -> {str(self.game._arena.final_state())}"
    
    def _observation(self) -> tuple:
        """RETURN OBSERVATION PARAMETERS"""
        sampling = self._range / 2

        on_ground = 1

        prev_top_platf = (float('inf'), float('inf'))
        next_top_platf = (float('inf'), float('inf'))

        prev_bot_platf = (float('inf'), float('inf'))
        next_bot_platf = (float('inf'), float('inf'))

        prev_ground_x = float('inf')
        next_ground_x = float('inf')

        coin = (float('inf'), float('inf'))
        plant = (float('inf'), float('inf'))


        # --- GET PLAYER INFORMATION ---

        for player in self.game._arena.actors():

            if isinstance(player, Player):
                
                on_ground = 1 if player.velocity() == 0 else 0
                
                for actor in self.game._arena.actors():
                    x_measure, y_measure = self.game._arena.actor_distance(player, actor)
                    
                    if math.sqrt(x_measure**2 + y_measure**2) < self._range:

                        xdist = int(sampling * (x_measure / self._range))
                        ydist = int(sampling * (y_measure / self._range))

                        if isinstance(actor, Platform) and actor.position()[1] < 400:
                            if x_measure >= 0:
                                next_top_platf = (xdist, ydist) if math.sqrt(xdist**2 + ydist**2) < math.sqrt(next_top_platf[0]**2 + next_top_platf[1]**2) else next_top_platf
                            elif x_measure < 0:
                                prev_top_platf = (abs(xdist), abs(ydist)) if math.sqrt(xdist**2 + ydist**2) < math.sqrt(prev_top_platf[0]**2 + prev_top_platf[1]**2) else prev_top_platf

                        if isinstance(actor, Platform) and actor.position()[1] >= 400:
                            if x_measure >= 0:
                                next_bot_platf = (xdist, ydist) if math.sqrt(xdist**2 + ydist**2) < math.sqrt(next_bot_platf[0]**2 + next_bot_platf[1]**2) else next_bot_platf
                            elif x_measure < 0:
                                prev_bot_platf = (abs(xdist), abs(ydist)) if math.sqrt(xdist**2 + ydist**2) < math.sqrt(prev_bot_platf[0]**2 + prev_bot_platf[1]**2) else prev_bot_platf

                        elif isinstance(actor, Ground):
                            if x_measure >= 0:
                                next_ground_x = xdist if xdist < next_ground_x else next_ground_x
                            elif x_measure < 0:
                                prev_ground_x = abs(xdist) if xdist < prev_ground_x else prev_ground_x

                        elif isinstance(actor, Coin):
                            coin = (xdist, ydist) if math.sqrt(xdist**2 + ydist**2) < math.sqrt(coin[0]**2 + coin[1]**2) else coin

                        elif isinstance(actor, Plant):
                            plant = (xdist, ydist) if math.sqrt(xdist**2 + ydist**2) < math.sqrt(plant[0]**2 + plant[1]**2) else plant

        return tuple([on_ground, prev_ground_x, next_ground_x, prev_top_platf, next_top_platf, prev_bot_platf, next_bot_platf, coin, plant])