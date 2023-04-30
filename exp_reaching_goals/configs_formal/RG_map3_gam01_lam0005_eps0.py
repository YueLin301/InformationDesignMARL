from utils.configdict import ConfigDict
from exp_reaching_goals.configs.path_config import config_path

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'RG_map3_gam01_lam0005_eps0'

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
# config.sender.gaussian_distribution = True

config.sender.lr_critic_Gi = 3e-4
config.sender.lr_critic_Gj = 3e-4
config.sender.lr_signal = 1.5e-4
config.sender.gamma = 0.99
config.sender.sender_objective_alpha = 0.005
config.sender.sender_constraint_right = 0
config.sender.coe_for_recovery_fromgumbel = 2
if config.sender.gaussian_distribution:
    config.sender.gaussian_var = 2
config.sender.epsilon_greedy = 0
config.sender.epsilon_decay = 0  # after generating every episode: epsilong <- epsilon * coe_decay
config.sender.epsilon_min = 0

# ==================================================
config.receiver = ConfigDict()
config.receiver.load = False
config.receiver.lr_actor = 3e-5
config.receiver.lr_critic_Gj = 3e-4
config.receiver.gamma = 0.1
# config.receiver.entropy_coe = 1e-4
config.receiver.entropy_coe = 0
config.receiver.obs_range = [1, 0]  # oj: receiver position, receiver's apple position

# ==================================================
config.n_channels = ConfigDict()
config.n_channels.obs_sender = 2 if config.env.aligned_object else 3  # receiver position, receiver apple position ; receiver position, sender apple position, receiver apple position
if config.sender.regradless_agent_pos:
    config.n_channels.obs_sender -= 1
config.n_channels.obs_and_message_receiver = sum(config.receiver.obs_range) + 1  # receiver position, message

# ==================================================
config.nn = ConfigDict()
config.nn.kernel = 5
config.nn.n_filters = 3
config.nn.hidden_width = 32
config.nn.stride = 1
config.nn.target_critic_tau = 0.98
config.nn.target_critic_howoften = 1
