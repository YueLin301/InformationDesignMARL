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

config.pro.lr_pro_critic = 2e-3
config.pro.lr_signal = 1e-3

config.pro.coe_for_recovery_fromgumbel = 2

config.pro.constraint_right = 0
config.pro.sender_objective_alpha = 2  # Lagrangian lambda

config.pro.initialize = False

# ========================================

'''hr'''
config.hr = ConfigDict()

config.hr.fixed_policy = False

config.hr.lr_critic = 5e-4
config.hr.lr_actor = 5e-4

config.hr.entropy_coe = 0

config.hr.initialize = False

# ========================================
'''train'''
config.train = ConfigDict()

config.train.n_episodes = 3e6

config.train.GAE_term = 'TD-error'
