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
config.pro.fixed_signaling_scheme = True
# optimal signaling scheme
config.pro.signaling_scheme = torch.tensor([[0.5, 0.5],
                                            [0, 1]], dtype=torch.double)
config.pro.lr_pro_critic = 0
config.pro.lr_signal = 0

config.pro.constraint_right = 0
config.pro.sender_objective_alpha = 0  # Lagrangian lambda

config.pro.initialize = False

# ========================================
'''hr'''
config.hr = ConfigDict()

config.hr.fixed_policy = False

config.hr.lr_critic = 1e-3
config.hr.lr_actor = 3e-3

config.hr.initialize = False

# ========================================
'''train'''
config.train = ConfigDict()

config.train.n_episodes = 5e5

config.train.GAE_term = 'TD-error'
# config.train.GAE_term = 'advantage'
