from utils.configdict import ConfigDict
from exp_recommendation.configs.env_config import config_env

config = ConfigDict()
# =========================================
'''env'''
config.env = config_env
# ========================================
'''pro'''
config.pro = ConfigDict()
config.pro.sender_objective_alpha = None
# ========================================
'''hr'''
config.hr = ConfigDict()
config.hr.lr_critic = 7e-4
config.hr.eps = 0.05
config.hr.gamma = 0.99
# ========================================
'''train'''
config.train = ConfigDict()
config.train.n_episodes = 3e6
