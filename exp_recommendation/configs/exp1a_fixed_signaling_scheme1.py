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
if config.pro.fixed_signaling_scheme:
    # 完全乱发消息
    config.pro.signaling_scheme = torch.tensor([[0.5, 0.5],
                                                [0.5, 0.5]], dtype=torch.double)
    config.pro.lr_pro_critic = 0
    config.pro.lr_signal = 0
else:
    config.pro.lr_pro_critic = 1e-3
    lr_signal = 1e-3

config.pro.constraint_right = 0  # 大于等于，改成等于，所以这个epsilon就不必了
config.pro.sender_objective_alpha = 0  # 这是拉格朗日的那个lambda

# ========================================
'''hr'''
config.hr = ConfigDict()
config.hr.lr_critic = 1e-3

config.hr.lr_actor = 3e-3

config.hr.epsilon_start = 0.5  # epsilon_greedy
config.hr.epsilon_decay = 0.999  # after running an episode: epsilon <- epsilon * epsilon_decay

# ========================================
'''train'''
config.train = ConfigDict()

config.train.initialize = False

# config.train.n_episodes = 3e4
config.train.n_episodes = 5e4

config.train.howoften_update = 5

config.train.GAE_term = 'TD-error'
