import torch
from utils.configdict import ConfigDict
from exp_recommendation.configs.env_config import config_env

config = ConfigDict()
# =========================================
'''env'''
config.env = config_env

# ========================================
'''pro'''
config.pro = ConfigDict()
config.pro.fixed_signaling_scheme = False
config.pro.initialize = False

config.pro.lr_pro_critic = 1e-3
config.pro.lr_signal = 1e-3

config.pro.constraint_right = 0
config.pro.sender_objective_alpha = 0.2  # Lagrangian lambda
# [0.2, 0.35]

# ========================================

'''hr'''
config.hr = ConfigDict()

config.hr.fixed_policy = False
config.hr.initialize = False

config.hr.lr_critic = 3e-3
config.hr.lr_actor = 5e-3

config.hr.epsilon_start = 0  # epsilon_greedy
config.hr.epsilon_decay = 0  # after running an episode: epsilon <- epsilon * epsilon_decay

# ========================================
'''train'''
config.train = ConfigDict()

# config.train.n_episodes = 3e4
# config.train.n_episodes = 6e4
# config.train.n_episodes = 1e5
config.train.n_episodes = 5e5

config.train.howoften_update = 20

config.train.GAE_term = 'TD-error'
