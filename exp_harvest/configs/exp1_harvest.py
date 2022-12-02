from utils.configdict import ConfigDict
from exp_harvest.configs.env_config import config_env

default_dim_action = 9

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'acb_learn'

# ==================================================
'''env'''
config.env = config_env

# ==================================================

config.sender = ConfigDict()
config.sender.n_channels = 3  # sender position, receiver position, apple position
config.sender.lr_actor = 1e-4
config.sender.lr_critic_Gi = 1e-3
config.sender.lr_critic_Gj = 1e-3
config.sender.lr_signal = 1e-3

config.sender.sender_objective_alpha = 0.2

# ==================================================

config.receiver = ConfigDict()
config.receiver.n_channels = 2  # receiver position, message(apple position)
config.receiver.lr_actor = 1e-4
config.receiver.lr_critic_Gj = 1e-3

# ==================================================

config.train = ConfigDict()
config.train.n_episodes = 200000
config.train.period = 500  # 多久保存一次
# config.train.n_episodes = 1
# config.train.period = 1

# ==================================================

config.alg = ConfigDict()
config.alg.epsilon_start = 0.5
config.alg.epsilon_end = 0.05
config.alg.epsilon_decay = 0.999  # 每次episode跑完，epsilon就乘这个数
config.alg.gamma = 0.99

# ==================================================
# 暂时不怎么需要改的
# ==================================================
config.nn = ConfigDict()
config.nn.kernel = 5
config.nn.n_filters = 3
config.nn.hidden_width = 32
config.nn.stride = 1
config.nn.target_critic_tau = 0.98
config.nn.target_critic_howoften = 1  # 过多少个episode更新一次target critic

# ==================================================
