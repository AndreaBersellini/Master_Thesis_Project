# MAP
ARENA_W, ARENA_H = (1000, 700)
LEVEL_LENGTH = 5000
GROUND_UNIT = 100
CELL_UNIT = 40
END = LEVEL_LENGTH - 300


# ACTORS
PALYER_SPEEDS = (10, 20, 0.9, 50)
PLAYER_W, PLAYER_H = 30, 60
COIN_W, COIN_H = 50, 50
PLANT_W, PLANT_H = 40, 60
GROUND_W, GROUND_H = 100, 70
PLATFORM_W, PLATFORM_H = 100, 50

ground_layout = [1] * int(LEVEL_LENGTH / GROUND_UNIT)
ground_holes = [8,11,12,16,20,21,22,23,28,29,32,33,34,35,36]
for i in ground_holes:
    ground_layout[i] = 0
print("GROUND VECTOR DIMENSION: " + str(len(ground_layout)))


platform_layout = []
platform_layout.append([0] * int(LEVEL_LENGTH / GROUND_UNIT))
platform_layout.append([0] * int(LEVEL_LENGTH / GROUND_UNIT))
platforms = [(0,9),(0,10),(0,12),(0,15),(0,21),(0,22),(0,25),(0,33),(0,35),
            (1,16),(1,17),(1,22),(1,23),(1,24),(1,34)]
for (i,j) in platforms:
    platform_layout[i][j] = 1
print("PLATFORM MATRIX DIMENSION: " + str(len(platform_layout)) + " x " + str(len(platform_layout[0])))


plant_layout = []
plant_layout.append([0] * int(LEVEL_LENGTH / CELL_UNIT))
plant_layout.append([0] * int(LEVEL_LENGTH / CELL_UNIT))
plants = [(0,43),(0,63),(0,78),(0,97),(0,114)]
for (i,j) in plants:
    plant_layout[i][j] = 1
print("PLANT MATRIX DIMENSION: " + str(len(plant_layout)) + " x " + str(len(plant_layout[0])))


coin_layout = []
for i in range(int(ARENA_H / CELL_UNIT)):
    coin_layout.append([0] * int(LEVEL_LENGTH / CELL_UNIT))
coins = [(14,24),(14,34),(14,94),(14,100),
        (12,42),(12,71),(12,114),
        (10,31),(10,53),(10,55),(10,85),
        (6,41),(6,43),(6,57),(6,59)]
for (i,j) in coins:
    coin_layout[i][j] = 1
print("COIN MATRIX DIMENSION: " + str(len(coin_layout)) + " x " + str(len(coin_layout[0])))