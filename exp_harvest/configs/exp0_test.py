from utils.configdict import ConfigDict
from exp_harvest.configs.env_config import config_env
from exp_harvest.configs.path_config import config_path

default_dim_action = 5

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'exp0_test'

# ==================================================
config.env = ConfigDict()
config.env = config_env

# ==================================================
config.path = ConfigDict()
config.path = config_path

# ==================================================

config.n_channels = ConfigDict()
config.n_channels.obs_sender = 3  # sender position, receiver position, apple position
config.n_channels.obs_receiver = 1  # receiver position
config.n_channels.message = 1  # fake apple position

# ==================================================

config.sender = ConfigDict()
config.sender.lr_actor = 1e-4
config.sender.lr_critic_Gi = 1e-3
config.sender.lr_critic_Gj = 1e-3
config.sender.lr_critic_foractor = 1e-3
config.sender.lr_signal = 1e-3

config.sender.gamma = 0.99
config.sender.entropy_coe = 1e-3

config.sender.sender_objective_alpha = 0.2

# ==================================================

config.receiver = ConfigDict()
config.receiver.lr_actor = 1e-4
config.receiver.lr_critic_Gj = 1e-3

config.receiver.gamma = 0.99
config.receiver.entropy_coe = 1e-3

# ==================================================

config.train = ConfigDict()
config.train.n_episodes = 10
config.train.period = 5  # 多久保存一次
config.train.buffer_size = 50

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
