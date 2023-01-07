from utils.configdict import ConfigDict
from exp_reaching_goals.configs.path_config import config_path

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'exp7b_map5_0005'

# ==================================================
config.env = ConfigDict()
config.env.map_height = 5
config.env.map_width = 5
config.env.max_step = 50
config.env.aligned_object = False
config.env.dim_action = 4
config.env.bounded = True

config.env.reward_amplifier = 30
config.env.punish_amplifier = 1

# ==================================================
config.train = ConfigDict()
config.train.n_episodes = 2000000
config.train.period = 500

# ==================================================
config.path = ConfigDict()
config.path = config_path

# ==================================================
config.sender = ConfigDict()
config.sender.honest = False
config.sender.regradless_agent_pos = False
config.sender.gaussian_distribution = False

config.sender.lr_critic_Gi = 3e-4
config.sender.lr_critic_Gj = 3e-4
config.sender.lr_signal = 1.5e-4
config.sender.gamma = 0.99
config.sender.sender_objective_alpha = 0.005
config.sender.coe_for_recovery_fromgumbel = 2
if config.sender.gaussian_distribution:
    config.sender.gaussian_var = 2
config.sender.epsilon_greedy = 0
config.sender.epsilon_decay = 0  # after generating every episode: epsilong <- epsilon * coe_decay
config.sender.epsilon_min = 0

# ==================================================
config.receiver = ConfigDict()
config.receiver.load = False
config.receiver.blind = False
config.receiver.lr_actor = 3e-5
config.receiver.lr_critic_Gj = 3e-4
config.receiver.gamma = 0.99
# config.receiver.entropy_coe = 1e-4
config.receiver.entropy_coe = 0

# ==================================================
config.n_channels = ConfigDict()
config.n_channels.obs_sender = 2 if config.env.aligned_object else 3  # receiver position, receiver apple position ; receiver position, sender apple position, receiver apple position
if config.sender.regradless_agent_pos:
    config.n_channels.obs_sender -= 1
config.n_channels.obs_and_message_receiver = 2 if not config.sender.honest else config.n_channels.obs_sender  # receiver position, fake apple position

# ==================================================
config.nn = ConfigDict()
config.nn.kernel = 5
config.nn.n_filters = 3
config.nn.hidden_width = 32
config.nn.stride = 1
config.nn.target_critic_tau = 0.98
config.nn.target_critic_howoften = 1
