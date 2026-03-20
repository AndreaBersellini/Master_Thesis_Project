from Visual_Deep_Q_Learning.agent import *
from Visual_Deep_Q_Learning.environment import *

project_name = 'Visual_Deep_Q_Learning'

# PATHS
# --------------------------------
paths = {
    'results' : os.path.join(project_name, 'checkpoints'),
    'policy_net' : os.path.join(project_name, 'checkpoints', '__policy_net.pth'),
    'target_net' : os.path.join(project_name, 'checkpoints', '__target_net.pth'),
    'params' : os.path.join(project_name, 'checkpoints', '__params.npy'),
    'replay_buff' : os.path.join(project_name, 'checkpoints', '__replay.npy')
}
# --------------------------------

# TRAINING ITERATIONS
# --------------------------------
iterations = {
    'generations' : 100,
    'episodes' : 20#200 # the test with epi=75 and target_upd=25 was promising
}
# --------------------------------

# HYPERPARAMETERS
# --------------------------------
epsilon_min = 0.01
epsilon_max = 0.5

hyperparameters = {
    'learning_rate' : 0.0005,
    'discount_factor' : 0.99,
    'batch_size' : 64,
    'buffer_size' : 1000000,
    'threshold' : 64
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
agent = AgentDQL(iterations, epsilon_min, epsilon_max, hyperparameters, paths, obs_space, act_space)
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