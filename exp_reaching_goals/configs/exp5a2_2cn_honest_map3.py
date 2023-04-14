from utils.configdict import ConfigDict
from exp_reaching_goals.configs.path_config import config_path

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'exp5a2_2cn_honest_map3'

# ==================================================
config.env = ConfigDict()
config.env.map_height = 3
config.env.map_width = 3
config.env.max_step = 50
config.env.aligned_object = False
config.env.dim_action = 4
config.env.bounded = True

config.env.reward_amplifier = 10
config.env.punish_amplifier = 3

# ==================================================
config.train = ConfigDict()
config.train.n_episodes = 200000
config.train.period = 500

# ==================================================
config.path = ConfigDict()
config.path = config_path

# ==================================================
config.sender = ConfigDict()

config.sender.honest = True
config.sender.regradless_agent_pos = False

# ==================================================
config.receiver = ConfigDict()
config.receiver.load = False
config.receiver.blind = False
config.receiver.blind = config.receiver.blind and config.sender.honest
config.receiver.lr_actor = 1e-4
config.receiver.lr_critic_Gj = 1e-3
config.receiver.gamma = 0.99
config.receiver.entropy_coe = 0

# ==================================================
config.n_channels = ConfigDict()
config.n_channels.obs_sender = 2 if config.env.aligned_object else 3  # receiver position, receiver apple position ; receiver position, sender apple position, receiver apple position
if config.sender.regradless_agent_pos:
    config.n_channels.obs_sender -= 1
config.n_channels.obs_and_message_receiver = 2  # receiver position, fake apple position

# ==================================================
config.nn = ConfigDict()
config.nn.kernel = 5
config.nn.n_filters = 3
config.nn.hidden_width = 32
config.nn.stride = 1
config.nn.target_critic_tau = 0.98
config.nn.target_critic_howoften = 1
