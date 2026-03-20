from .classes import *
from .map_parameters import *

class SuperMarion():
    def __init__(self, mode : str):
       
        self._mode = mode  # MODE (USER / AGENT)

        # ARENA SETUP
        self._arena = Arena((ARENA_W, ARENA_H), self._mode)

        # PROGRESS SETUP
        progress = Progress(self._arena, (ARENA_W / 2 + 30, 0), (5, ARENA_H))
        self._arena.spawn(progress)

        # FINISH LINE
        flag = Flag(self._arena, (END, 0), (50, ARENA_H))
        self._arena.spawn(flag)

        # GROUND LAYOUT
        for i, g in enumerate(ground_layout):
            if g:
                ground = Ground((i * GROUND_UNIT, ARENA_H - GROUND_H))
                self._arena.spawn(ground)
        
        # PLATFORM LAYOUT
        for i, row in enumerate(platform_layout):
            for j, c in enumerate(row):
                if c:
                    platform = Platform((j * GROUND_UNIT, ARENA_H - 220 - i * 150))
                    self._arena.spawn(platform)

        # PLANT LAYOUT
        for i, row in enumerate(plant_layout):
            for j, c in enumerate(row):
                if c:
                    plant = Plant(self._arena, (j * CELL_UNIT, ARENA_H - 130 - i * 150))
                    self._arena.spawn(plant)
                
        # COIN LAYOUT
        for i, row in enumerate(coin_layout):
            for j, c in enumerate(row):
                if c:
                    coin = Coin(self._arena, (j * CELL_UNIT, i * CELL_UNIT))
                    self._arena.spawn(coin)

        #self._arena.spawn(Coin(self._arena, (20 * CELL_UNIT, 12 * CELL_UNIT))) # <---- DEBUG
        
        # PLAYER
        player = Player(self._arena, (ARENA_W / 2, ARENA_H / 2), PALYER_SPEEDS)
        self._arena.spawn(player)

    def tick(self):
        """GAME TICK"""

        g2d.clear_canvas()
        self._arena.draw()
        self._arena.tick(g2d.current_keys())

    def start(self) -> None:
        """START THE MAIN LOOP ONLY IN USER MODE"""

        g2d.init_canvas((ARENA_W, ARENA_H))
        g2d.main_loop(self.tick)