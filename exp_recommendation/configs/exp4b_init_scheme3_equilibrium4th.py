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

# config.pro.coe_for_recovery_fromgumbel = 1
config.pro.coe_for_recovery_fromgumbel = 1.7

config.pro.constraint_right = 0
# config.pro.sender_objective_alpha = 1  # Lagrangian lambda
# config.pro.sender_objective_alpha =  10  # Lagrangian lambda
# config.pro.sender_objective_alpha = 50  # Lagrangian lambda
# config.pro.sender_objective_alpha = 100  # Lagrangian lambda
config.pro.sender_objective_alpha = 200  # Lagrangian lambda

config.pro.initialize = True
if config.pro.initialize:
    # signaling scheme of pro is initialized to be optimal
    config.pro.signaling_params = torch.tensor([[10, 0],
                                                [10, 10]], dtype=torch.double).unsqueeze(dim=0)

# ========================================

'''hr'''
config.hr = ConfigDict()

config.hr.fixed_policy = False

config.hr.lr_critic = 2e-3
config.hr.lr_actor = 2e-3

config.hr.entropy_coe = 1e-3

config.hr.initialize = True
if config.hr.initialize:
    # policy of hr is initialized to be obedient
    config.hr.actor_params = torch.tensor([[5, -5],
                                           [-5, 5]], dtype=torch.double).unsqueeze(dim=0)

# ========================================
'''train'''
config.train = ConfigDict()

config.train.n_episodes = 1e6

config.train.GAE_term = 'TD-error'