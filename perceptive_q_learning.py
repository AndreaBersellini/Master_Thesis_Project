from Perceptive_Q_Learning.agent import *
from Perceptive_Q_Learning.environment import *

project_name = 'Perceptive_Q_Learning'

# PATHS
# --------------------------------
paths = {
    'results' : os.path.join(project_name, 'checkpoints'),
    'table' : os.path.join(project_name, 'checkpoints', '__table.npy'),
    'params' : os.path.join(project_name, 'checkpoints', '__params.npy')
}
# --------------------------------

# TRAINING ITERATIONS
# --------------------------------
iterations = {
    'generations' : 500,
    'episodes' : 200,
}
# --------------------------------

# HYPERPARAMETERS
# --------------------------------
epsilon_min = 0.05
epsilon_max = 0.6

hyperparameters = {
    'learning_rate' : 0.05,
    'discount_factor' : 0.9
}
# --------------------------------

# ENVIRONMENT
# --------------------------------
env = CustomEnv()
obs_space = env.obs_space_size
act_space = env.act_space_size
# --------------------------------

# AGENT
# --------------------------------
agent = AgentQL(iterations, epsilon_min, epsilon_max, hyperparameters, paths, obs_space, act_space)
# --------------------------------

while True:
    i = input("Train the model? 'y' or 'n': ").strip().lower()
    match i:
        case 'y':
            agent.train(env)
            break
        case 'n':
            time.sleep(3)
            print(agent.test(env))
            break
        case _:
            print("Invalid input!")
