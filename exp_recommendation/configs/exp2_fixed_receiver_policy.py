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

config.pro.lr_pro_critic = 1.5e-3
config.pro.lr_signal = 1.5e-3

config.pro.coe_for_recovery_fromgumbel = 2

config.pro.constraint_right = 0
config.pro.sender_objective_alpha = 0  # Lagrangian lambda

config.pro.initialize = False

# ========================================
'''hr'''
config.hr = ConfigDict()

config.hr.fixed_policy = True

config.hr.lr_critic = 0
config.hr.lr_actor = 0

config.hr.entropy_coe = 0

config.hr.initialize = True
if config.hr.initialize:
    # policy of hr is initialized to be obedient
    config.hr.actor_params = torch.tensor([[10, -10],
                                           [-10, 10]], dtype=torch.double).unsqueeze(dim=0)

# ========================================
'''train'''
config.train = ConfigDict()

# config.train.n_episodes = 3e4
# config.train.n_episodes = 6e4
config.train.n_episodes = 3e5

config.train.GAE_term = 'TD-error'
