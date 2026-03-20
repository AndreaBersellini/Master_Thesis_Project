import os, sys, time, keyboard
from abc import ABC, abstractmethod

from . import g2d

Point = tuple[float, float]

class Actor(ABC):

    @abstractmethod
    def relative_move(self):
        pass
    
    @abstractmethod
    def absolute_move(self, distance : float) -> None:
        pass
    
    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def position(self) -> Point:
        pass

    @abstractmethod
    def size(self) -> Point:
        pass

def get_image_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def check_collision(a1: Actor, a2: Actor) -> bool:
    x1, y1, w1, h1 = a1.position() + a1.size()
    x2, y2, w2, h2 = a2.position() + a2.size()
    return (y2 <= y1 + h1 and y1 <= y2 + h2 and
            x2 <= x1 + w1 and x1 <= x2 + w2)

class Arena:
    def __init__(self, size : 'Point', mode : str):
        # MODE
        self._mode = mode

        # DIMENTIONS
        self._w, self._h = size
        self._level_lenght = 5000
        self._bg_x = 0

        # ACTORS
        self._actors: list[Actor] = []
        
        # GAME STATE
        self._curr_keys = self._prev_keys = tuple()
        self._collisions = []
        self._turn = -1
        self._final_state = 0

        # SCORE AND METRICS
        self._player_score = 0
        self._max_distance = 0
        self._frame_count = 0

        self._menu = Menu(self)
        self._finish = EndScreen(self)

    def _naive_collisions(self, actors : list) -> None:
        # self._collisions = [[a2 for a2 in actors if a1 is not a2 and check_collision(a1, a2)] for a1 in actors]
        self._collisions.clear()
        for a1 in actors:
            colls1 = []
            for a2 in actors:
                if a1 is not a2 and check_collision(a1, a2):
                    colls1.append(a2)
            self._collisions.append(colls1)

    def _detect_collisions(self, actors : list) -> None:
        self._collisions.clear()
        tile = 40
        nx, ny = -(-self._w // tile),  -(-self._h // tile)
        cells = [set() for _ in range(nx * ny)]
        for i, a in enumerate(actors):
            x, y, w, h = (round(v) for v in a.position() + a.size())
            for tx in range((x - 1) // tile, 1 + (x + w + 1) // tile):
                for ty in range((y - 1) // tile, 1 + (y + h + 1) // tile):
                    if 0 <= tx < nx and 0 <= ty < ny:
                        cells[ty * nx + tx].add(i)
        for i, a in enumerate(actors):
            neighs = set()
            x, y, w, h = (round(v) for v in a.position() + a.size())
            for tx in range((x - 1) // tile, 1 + (x + w + 1) // tile):
                for ty in range((y - 1) // tile, 1 + (y + h + 1) // tile):
                    if 0 <= tx < nx and 0 <= ty < ny:
                        neighs |= cells[ty * nx + tx]
            colls = [actors[j] for j in sorted(neighs, reverse=True)
                     if i != j and check_collision(a, actors[j])]
            self._collisions.append(colls)

    def spawn(self, a: Actor) -> None:
        # ADD A NEW ACTOR TO THE ARENA
        if a not in self._actors:
            self._actors.append(a)

    def kill(self, a: Actor) -> None:
        # REMOVE ACTOR FROM ARENA
        if a in self._actors:
            self._actors.remove(a)

        # REMOVE PLAYER AND END THE GAME
        if isinstance(a, Player):
            self.end_game("DEFEAT")

    def tick(self, keys=[]) -> None:
        actors = list(reversed(self._actors))

        # DETECT COLLISIONS BETWEEN ACTORS
        self._detect_collisions(actors)

        # DETECT KEYBOARD INPUT
        self._prev_keys = self._curr_keys
        self._curr_keys = keys

        # INTERPRET KEYBOARD INPUT
        if self._mode == "USER":
            # RENDER MENU
            if self._menu.active():
                self._menu.draw()

            if "Escape" in self._curr_keys and "Escape" not in self._prev_keys:
                self._menu.draw()
                self._menu.toggle(True)

            # RENDER FINISH SCREEN
            if self._finish.active():
                self._finish.draw()

        # MOVE EACH ACTORS
        for self._turn, a in enumerate(actors):
            a.relative_move()

        for a in self._actors:
            if isinstance(a, Progress):
                self._max_distance = a.distance()

        # INCREMENT FRAME COUNT
        self._frame_count += 1

    def draw(self) -> None:
        # RENDER BACKGROUND
        for i in range(int(self._level_lenght / self._w)):
            g2d.draw_image(get_image_path('images/background.png'), (self._bg_x + i * self._w, 0), (0, 0), (self._w, self._h))

        # RENDER TOP BAR
        g2d.set_color((50, 50, 50))
        g2d.draw_rect((0, 0), (self._w, 85))
        g2d.set_color((150, 150, 160))
        g2d.draw_rect((0, 0), (self._w, 80))

        # RENDER SCORE
        g2d.set_color((30, 30, 30))
        g2d.draw_text("SCORE: " + str(self._player_score), (80, 40), 20)

        # RENDER DISTANCE
        g2d.draw_text("DISTANCE: " + str(self._max_distance), (self._w / 2, 40), 20)

        # RENDER TIME
        g2d.draw_text("TIME: " + str(time.strftime('%M:%S:', time.gmtime(self._frame_count / 30))) + str(int(self._frame_count / 30 * 10) % 10), (self._w - 80, 40), 20)

        # RENDER ACTORS
        for a in self._actors:
            a.draw()

    def control(self, action : int) -> None:
        # FORCE INPUT OF A PLAYER ACTION
        for a in self._actors:
            if isinstance(a, Player):
                match action:
                    case 0:
                        pass # Do Nothing
                    case 1:
                        a.add_key("a") # Move left
                    case 2:
                        a.add_key("d") # Move right
                    case 3:
                        a.add_key("Spacebar")  # Jump
                    case _:
                        raise Exception(f"Invalid action. Action {action} cannot be executed by the player!")

    def move_actors(self, distance : float, px : float, pw : float) -> None:
        actors = list(reversed(self._actors))

        # CHECK THE LEFT AND RIGHT BORDER OF THE LEVEL
        stop_left = self._bg_x == 0 and (px - distance) <= (self._w / 2)
        stop_right = self._bg_x == (-self._level_lenght + self._w) and (px + pw - distance) >= (self._w / 2)
        
        # MOVE THE ACTORS BASED ON ABSOLUTE POSITION
        if stop_left or stop_right:
            for a in actors:
                if isinstance(a, Player):
                    a.absolute_move(-distance)
        elif self._bg_x + distance <= 0 and self._bg_x + distance >= -self._level_lenght + self._w:
            for a in actors:
                if not isinstance(a, Player):
                    a.absolute_move(distance)
            self._bg_x += distance
        else:
            if self._bg_x + distance > 0:
                self._bg_x = 0 # COMPENSATION IF THE RESULT OF BG + DIST IS NOT 0
            elif self._bg_x + distance < (-self._level_lenght + self._w):
                self._bg_x = (-self._level_lenght + self._w) # COMPENSATION IF THE RESULT OF BG + DIST IS NOT DIVISOR OF LEV_LEN + ARENA_W

    def increase_score(self, value: int) -> None:
        self._player_score += value

    def end_game(self, state : str) -> None:
        # SET THE FINAL GAME STATE
        if state == "VICTORY":
            self._final_state = 1
        elif state == "DEFEAT":
            self._final_state = 2

        # RENDER THE GAME OVER MENU
        if self._mode == "USER":
            stats = self.metrics()
            stats['state'] = state
            self._finish.set_stats(stats)
            self._finish.draw()
            self._finish.toggle(True)

    def actors(self) -> list:
        return list(self._actors)

    def collisions(self) -> list[Actor]:
        t, colls = self._turn, self._collisions
        return colls[t] if 0 <= t < len(colls) else []

    def current_keys(self) -> list[str]:
        return self._curr_keys

    def previous_keys(self) -> list[str]:
        return self._prev_keys

    def size(self) -> 'Point':
        return (self._w, self._h)

    def frame_count(self) -> int:
        return self._frame_count

    def final_state(self) -> int:
        return self._final_state
    
    def actor_distance(self, a1 : 'Actor', a2 : 'Actor') -> tuple[float, float]:
        x1, y1 = a1.position()
        x2, y2 = a2.position()

        return (x2 - x1, y2 - y1)

    def game_running(self) -> bool:
        # CHECK IF THE GAME IS RUNNING
        return True if self._final_state == 0 else False

    def metrics(self) -> dict:
        return {'score': self._player_score, 'total_distance': self._max_distance, 'frames': self._frame_count}
    
class Ground(Actor):
    def __init__(self, pos : 'Point'):
        self._x, self._y = pos
        self._w, self._h = 100, 70

    def absolute_move(self, distance : float) -> None:
        self._x += distance

    def relative_move(self) -> None:
        pass

    def draw(self) -> None:
        position = (self._x, self._y)
        size = (self._w, self._h)
        g2d.draw_image(get_image_path('images/ground_100.png'), position, (0, 0), size)

    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h

class Platform(Actor):
    def __init__(self, pos : 'Point'):
        self._x, self._y = pos
        self._w, self._h = 100, 50

    def absolute_move(self, distance : float) -> None:
        self._x += distance

    def relative_move(self) -> None:
        pass
    
    def draw(self) -> None:
        position = (self._x, self._y)
        size = (self._w, self._h)
        g2d.draw_image(get_image_path('images/platform_100.png'), position, (0, 0), size)

    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h

class Player(Actor):
    def __init__(self, arena : 'Arena', pos : 'Point', spd : tuple):
        # ARENA
        self._arena = arena

        # POSITION
        self._x, self._y = pos
        self._w, self._h = 30, 60
        self._absolute_pos = self._x
        self._prev_pos = (self._y, self._y + self._h)

        # SPEED
        self._speed_x = spd[0]
        self._speed_y = spd[1]
        self._gravity = spd[2]
        self._max_speed = spd[3]

        # STATE
        self._life = 1
        self._pose = 1

        # AGENT ACTIONS
        self._autoplay_keys = []

    def force_start(self, x : int, y : int) -> None:
        delta_x = x - self._absolute_pos
        self._absolute_pos += delta_x
        self._arena.move_actors(-delta_x, self._x, self._w)
        self._y = y

    def absolute_move(self, distance : float) -> None:
        self._x += distance

    def relative_move(self) -> None:
        arena_w, arena_h = self._arena.size()
        prev1, prev2 = self._prev_pos
        stop = False
        ground = False

        # COLLISIONS
        for other in self._arena.collisions():
            ox, oy = other.position()
            _, oh = other.size()

            if isinstance(other, Ground):
                if prev2 <= oy:
                    self._y = oy - self._h
                    self._speed_y = 0
                    ground = True
                elif self._x < ox:
                    stop = True
                elif self._x > ox:
                    stop = True

            if isinstance(other, Platform):
                if prev2 <= oy:
                    self._y = oy - self._h
                    self._speed_y = 0
                    ground = True
                elif prev1 >= oy + oh:
                    self._y = oy + oh
                    self._speed_y = -self._speed_y
                elif self._x < ox:
                    stop = True
                elif self._x > ox:
                    stop = True

        # CONTROLS
        keys = self._arena.current_keys()
        prev_keys = self._arena.previous_keys()
        
        # AUTOPLAY KEYS
        for key in self._autoplay_keys:
            keys.append(key)

        # CONTROL KEYS
        if "Spacebar" in keys and "Spacebar" not in prev_keys:
            if self._speed_y == 0: # !!! MAYBE EXPLOIT ALLOWS DOUBLE JUMP(?) !!!
                self._speed_y = -17 # JUMP SPEED
        if "a" in keys:
            if not stop:
                self._absolute_pos += -self._speed_x
                self._arena.move_actors(self._speed_x, self._x, self._w)
                if self._pose < 9: self._pose += 1
                else: self._pose = 0
        if "d" in keys:
            if not stop:
                self._absolute_pos += self._speed_x
                self._arena.move_actors(-self._speed_x, self._x, self._w)
                if self._pose < 9: self._pose += 1
                else: self._pose = 0

        self._autoplay_keys.clear() # RESET AUTO INPUT LIST
        
        if ground == False : self._speed_y += self._gravity 

        if self._x < 0:
            self._x = 0
        elif self._x + self._w > arena_w:
            self._x = arena_w - self._w

        if self._y >= arena_h:
            self._life = 0

        if self._life == 0:
            self._arena.kill(self)

        # MOVEMENT
        self._speed_y = self._max_speed if self._speed_y > self._max_speed else self._speed_y
        self._speed_y = -self._max_speed if self._speed_y < -self._max_speed else self._speed_y

        self._prev_pos = (self._y, self._y + self._h)
        self._y += self._speed_y

    def draw(self) -> None:
        position = (self._x, self._y)
        size = (self._w, self._h)

        if self._speed_y != 0:
            g2d.draw_image(get_image_path('images/player_jump.png'), position, (0, 0), size)
        elif self._pose < 4:
            g2d.draw_image(get_image_path('images/player_00.png'), position, (0, 0), size)
        else:
            g2d.draw_image(get_image_path('images/player_01.png'), position, (0, 0), size)

        #g2d.set_color((255, 30, 30))
        #g2d.draw_circle(position, 500)

    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h
    
    def add_key(self, key : str) -> None:
        self._autoplay_keys.append(key)
    
    def absolute_position(self) -> float:
        return self._absolute_pos
    
    def velocity(self) -> float:
        return self._speed_y

class Coin(Actor):
    def __init__(self, arena : 'Arena', pos : 'Point'):
        self._arena = arena
        self._x, self._y = pos
        self._w, self._h = 50, 50
        self._value = 1

    def absolute_move(self, distance : float) -> None:
        self._x += distance

    def relative_move(self) -> None:
        # COLLISIONS
        for other in self._arena.collisions():
            if isinstance(other, Player):
                self._arena.kill(self)
                self._arena.increase_score(self._value)

    def draw(self) -> None:
        position = (self._x, self._y)
        size = (self._w, self._h)
        g2d.draw_image(get_image_path('images/coin.png'), position, (0, 0), size)
    
    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h

class Progress(Actor):
    def __init__(self, arena : 'Arena', pos : 'Point', size : 'Point'):
        self._arena = arena
        self._x, self._y = pos
        self._w, self._h = size
        self._distance = 0

    def distance(self) -> float:
        return self._distance

    def absolute_move(self, distance : float) -> None:
        self._x += distance

    def relative_move(self) -> None:
        # COLLISIONS
        for other in self._arena.collisions():
            if isinstance(other, Player):
                px, _ = other.position()
                pw, _ = other.size()
                if self._x < px + pw:
                    self._distance += ((px + pw) - self._x)
                    self._x = px + pw

    def draw(self) -> None:
        #g2d.set_color((30, 30, 30))
        #g2d.draw_rect(self.position(), self.size())
        pass
    
    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h

class Flag(Actor):
    def __init__(self, arena : 'Arena', pos : 'Point', size : 'Point'):
        self._arena = arena
        self._x, self._y = pos
        self._w, self._h = size

    def absolute_move(self, distance : float) -> None:
        self._x += distance

    def relative_move(self) -> None:      
        # COLLISIONS
        for other in self._arena.collisions():
            if isinstance(other, Player):
                ox, oy = other.position()
                if ox >= self._x + self._w / 2:
                    self._arena.end_game("VICTORY")


    def draw(self) -> None:
        position = (self._x, self._y)
        size = (self._w, self._h)
        g2d.draw_image(get_image_path('images/flag.png'), position, (0, 0), size)
    
    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h

class Plant(Actor):
    def __init__(self, arena : 'Arena', pos : 'Point'):
        self._arena = arena
        self._x, self._y = pos
        self._w, self._h = 40, 60
        self._active = True

    def absolute_move(self, distance : float) -> None:
        self._x += distance

    def relative_move(self) -> None:      
        # COLLISIONS
        for other in self._arena.collisions():
            if isinstance(other, Player):
                if self._active:
                    self._active = False
                    self._arena.kill(other)

    def draw(self) -> None:
        position = (self._x, self._y)
        size = (self._w, self._h)
        if self._active:
            g2d.draw_image(get_image_path('images/plant_0.png'), position, (0, 0), size)
        else:
            g2d.draw_image(get_image_path('images/plant_1.png'), position, (0, 0), size)
    
    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h

class Menu(Actor):
    def __init__(self, arena : 'Arena'):
        self._arena = arena

        aw, ah = self._arena.size()
        self._x, self._y = aw / 2 - 300, ah / 2 - 200
        self._w, self._h = 600, 400
        self._show = False

    def active(self) -> bool:
        return self._show

    def toggle(self, state : bool) -> None:
        self._show = state

    def absolute_move(self, distance : float) -> None:
        pass

    def relative_move(self) -> None:      
        pass
        
    def draw(self) -> None:
        position = (self._x, self._y)
        size = (self._w, self._h)
        g2d.draw_image(get_image_path('images/pause_menu.png'), position, (0, 0), size)
        #g2d.set_color((30, 30, 30))
        #g2d.draw_rect(position, size)
        if self._show:
            while self._show:
                key = keyboard.read_key()
                match key:
                    case "f":
                        self.toggle(False)
                    case "g":
                        exit(1)
                    
    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h

class EndScreen(Actor):
    def __init__(self, arena : 'Arena'):
        self._arena = arena

        aw, ah = self._arena.size()
        self._x, self._y = aw / 2 - 300, ah / 2 - 200
        self._w, self._h = 600, 400
        self._show = False
        self._stats = {}

    def set_stats(self, statisctics : dict) -> None:
        self._stats = statisctics

    def active(self) -> bool:
        return self._show

    def toggle(self, state : bool) -> None:
        self._show = state

    def absolute_move(self, distance : float) -> None:
        pass

    def relative_move(self) -> None:      
        pass
        
    def draw(self) -> None:
        position = (self._x, self._y)
        size = (self._w, self._h)
        g2d.draw_image(get_image_path('images/pause_menu.png'), position, (0, 0), size)
        g2d.set_color((30, 30, 30))

        g2d.draw_text(self._stats['state'], (self._x + self._w / 2, self._y + 40), 30)
        g2d.draw_text(f"Total coins: {self._stats['score'] / 10}", (self._x + self._w / 2, self._y + 80), 30)
        g2d.draw_text(f"Max distance: {self._stats['total_distance']}", (self._x + self._w / 2, self._y + 120), 30)
        g2d.draw_text(f"Frames: {self._stats['frames']}", (self._x + self._w / 2, self._y + 160), 30)

        #g2d.draw_rect(position, size)
        if self._show:
            while self._show:
                key = keyboard.read_key()
                match key:
                    case "q":
                        os.execl(sys.executable, sys.executable, *sys.argv)
                    case "e":
                        exit(1)
                    
    def position(self) -> 'Point':
        return self._x, self._y

    def size(self) -> 'Point':
        return self._w, self._h
    
"""
class Sensor(Actor):
    def __init__(self, arena : 'Arena', pos : 'Point', radius : float):
        self._arena = arena
        self._x, self._y = pos

        self._x -= radius
        self._y -= radius

        self._w, self._h = radius * 2, radius * 2
        self._radius = radius

        self._perception = [Actor]

    def absolute_move(self, distance : float) -> None:
        self._x += distance

    def relative_move(self, pos : 'Point') -> None:      
        # COLLISIONS
        for other in self._arena.collisions():
            if isinstance(other, Platform):
                other.position()
                pass
        
        self._x, self._y = pos

    def draw(self) -> None:
        position = (self._x - self._radius, self._y - self._radius)
        size = (self._w, self._h)
        g2d.draw_rect(position, size)
    
    def position(self) -> 'Point':
        return self._x - self._radius, self._y - self._radius

    def size(self) -> 'Point':
        return self._w , self._h
"""