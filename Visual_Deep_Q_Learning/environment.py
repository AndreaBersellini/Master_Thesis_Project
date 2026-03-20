import cv2
import numpy as np
from XShared.environment.game.super_marion import *
from XShared.environment.env_abstract import Environment
from PIL import Image
import random
class CustomEnv(Environment):
    def __init__(self):
        super().__init__()

        self.act_space_size = 4
        self.obs_space_size = 84
        self._range = 500

        self._rewards = {
            'death' : -1,
            'victory' : 1,
            'progress' : 0.01,
            'coin' : 1,
            'stall' : -0.01
        }

        self._prev_p_pos = (0, 0)

    def _crop_image(self, image : np.ndarray) -> np.ndarray:
        h, w = image.shape
        x, y = self._prev_p_pos

        for a in self.game._arena.actors():
            if isinstance(a, Player):
                (px, py), (sx, sy) = a.position(), a.size()
                x, y = (int(px + sx / 2 - self._range / 2), int(py + sy / 2 - self._range / 2))
                self._prev_p_pos = (x, y)
                
        cropped = np.full((self._range, self._range), 0, dtype=image.dtype)

        x1_img = max(x, 0)
        y1_img = max(y, 0)
        x2_img = min(x + self._range, w)
        y2_img = min(y + self._range, h)

        x1_crop = x1_img - x
        y1_crop = y1_img - y
        x2_crop = x1_crop + (x2_img - x1_img)
        y2_crop = y1_crop + (y2_img - y1_img)

        cropped[y1_crop:y2_crop, x1_crop:x2_crop] = image[y1_img:y2_img, x1_img:x2_img]

        return cropped
    
    def _process_image(self, pixel_array : np.ndarray) -> np.ndarray:

        image = np.transpose(pixel_array, (1, 0, 2))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self._crop_image(image)
        image = cv2.resize(image, (84, 84))
        #cv2.namedWindow("Window Title", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Window Title", 300, 300)
        #cv2.imshow("Window Title", image)
        #key = cv2.waitKey(0)
        #if key == ord('h'):
        #    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #    pil_image = Image.fromarray(image_rgb)
        #    pil_image.save(f"output{random.randint(0,100)}.pdf")
        image = image / 255.0
        image = np.array([image])
        #print(image.shape)
        
        return image

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

        try:
            pixel_array = self._current_frame()
            state = self._process_image(pixel_array)
        except Exception as e:
            print(f"Error during image processing: {e}")
            #print(pixel_array)
            image = np.transpose(pixel_array, (1, 0, 2))
            cv2.namedWindow("Error", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Error", 300, 300)
            cv2.imshow("Error", image)
            cv2.waitKey(0)
        #print(state)
        #time.sleep(1)
        return state
    
    def render(self) -> None:
        self._draw()
        self._update()