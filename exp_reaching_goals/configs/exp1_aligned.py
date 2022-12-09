from utils.configdict import ConfigDict
from exp_reaching_goals.configs.path_config import config_path

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'exp1_aligned'

# ==================================================
config.env = ConfigDict()
config.env.map_height = 7
config.env.map_width = 7
config.env.max_step = 50
config.env.aligned_object = True

# ==================================================
config.train = ConfigDict()
config.train.batch_size = config.env.max_step - 1
config.train.n_episodes = 50000
config.train.period = 500

# ==================================================
config.path = ConfigDict()
config.path = config_path

# ==================================================
config.n_channels = ConfigDict()
config.n_channels.obs_sender = 2 if config.env.aligned_object else 3  # receiver position, sender apple position, receiver apple position
config.n_channels.message = 1  # fake apple position

# ==================================================
config.sender = ConfigDict()
config.sender.lr_critic_Gi = 1e-3
config.sender.lr_critic_Gj = 1e-3
config.sender.lr_signal = 1e-3
config.sender.gamma = 0.99
config.sender.sender_objective_alpha = 0.2
config.sender.coe_for_recovery_fromgumbel = 1  # TODO

# ==================================================
config.receiver = ConfigDict()
config.receiver.lr_actor = 1e-4
config.receiver.lr_critic_Gj = 1e-3
config.receiver.gamma = 0.99
config.receiver.entropy_coe = 1e-4

# ==================================================
config.nn = ConfigDict()
config.nn.kernel = 5
config.nn.n_filters = 3
config.nn.hidden_width = 32
config.nn.stride = 1
config.nn.target_critic_tau = 0.98
config.nn.target_critic_howoften = 1  # 过多少个episode更新一次target critic
