from utils.configdict import ConfigDict
from exp_reaching_goals.configs.path_config import config_path

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'RG_map3_DIAL'

# ==================================================
config.env = ConfigDict()
config.env.map_height = 3
config.env.map_width = 3
config.env.max_step = 50
config.env.aligned_object = False
config.env.dim_action = 4
config.env.bounded = True

config.env.reward_amplifier = 20
config.env.punish_amplifier = 5

# ==================================================
config.train = ConfigDict()
config.train.n_episodes = 100000
config.train.n_episodes *= 2
config.train.period = 500
# config.train.n_episodes = 50
# config.train.period = 25

# ==================================================
config.path = ConfigDict()
config.path = config_path

# ==================================================
config.sender = ConfigDict()
config.sender.honest = False
config.sender.regradless_agent_pos = False
config.sender.gaussian_distribution = False
config.sender.lr_critic = 1.5e-4
config.sender.sender_objective_alpha = 'DIAL'
config.sender.epsilon_decay = 0
config.sender.epsilon_min = 0

# ==================================================
config.receiver = ConfigDict()
config.receiver.load = False
config.receiver.blind = False
config.receiver.lr_critic = 3e-4
config.receiver.eps = 0.05
config.receiver.gamma = 0.1
# config.receiver.entropy_coe = 1e-4
config.receiver.entropy_coe = 0
config.receiver.obs_range = [1, 0]  # oj: receiver position, receiver's apple position

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
