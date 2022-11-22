'''
第1个字母表示地图
地图a：7*7原始地图，LIO里的小地图

第2个字母表示agent的setting
a: 1 agent
b: 2 agents
c: 2 agents purple sender
d: 2 agents blue sender

第3个字母表示agent是否看得到东西
a: 看得到
b: 看不到

'''

from utils.configdict import ConfigDict

default_dim_action = 9

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'acb_learn'

# ==================================================

config.env = ConfigDict()
config.env.name = 'clean up'
config.env.map_name = 'CLEANUP_SMALL_same_col'
config.env.n_agents = 2
config.env.is_sender_list = [True, False]
config.env.blind_receivers = True
config.env.signaling_objective_table = [[0],
                                        None]  # 只有0号是sender，且其只优化1号的收益期望
config.env.fixed_signaling_scheme = False

config.env.obs_height = 5  # obs的左右总长度，必须是奇数，viewsize是其-1的一半
config.env.obs_width = config.env.obs_height
config.env.global_ref_point = [3, 3]  # (7-1)/2 if not None, a fixed global reference frame is used for all agents
config.env.max_steps = 50

config.env.random_orientation = False
config.env.shuffle_spawn = False
config.env.disable_left_right_action = False
config.env.disable_rotation_action = True
config.env.cleaning_penalty = 0.0

config.env.cleanup_params = ConfigDict()
config.env.cleanup_params.appleRespawnProbability = 0.5  # 10x10 0.3 | small 0.5
config.env.cleanup_params.wasteSpawnProbability = 0.5  # 10x10 0.5 | small 0.5
config.env.cleanup_params.thresholdDepletion = 0.6  # 10x10 0.4 | small 0.6
config.env.cleanup_params.thresholdRestoration = 0.0  # 10x10 0.0 | small 0

config.env.use_agent_position_channel = True
config.env.use_apple_position_channel = True
config.env.use_waste_position_channel = True

# ==================================================

config.nn = ConfigDict()
config.nn.actor_howmany_linear = 4
config.nn.critic_howmany_linear = 4
config.nn.signaling_net_howmany_linear = 4

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
config.alg.lr_actor = 1e-4
config.alg.lr_critic = 1e-3
config.alg.lr_signal = 1e-3
config.alg.optimizer = 'adam'
config.alg.sender_objective_alpha = 0.2

# ==================================================
# 暂时不怎么需要改的
# ==================================================

config.nn.kernel = 5
config.nn.n_filters = 3
# config.nn.n_h1 = 32
config.nn.n_h2 = 32
config.nn.stride = 1
config.nn.rnn_required = False
config.nn.rnnGRU_hiddennum = 16
config.nn.target_critic_tau = 0.98
config.nn.target_critic_howoften = 1  # 过多少个episode更新一次target critic

config.env.beam_width = 3
config.env.n_senders = sum(config.env.is_sender_list)
config.env.obs_isimage = True  # 观测设置只有两种，一种是图片（单通道、多通道，多通道可以选择手动merge彩色转灰度，或者是自己学习merge），另一种是一维的向量
config.nn.conv_required = config.env.obs_isimage
config.env.obs_vector = config.env.obs_height * config.env.obs_width * config.nn.n_filters  # flatten成一维向量
# config.env.obs_vector = config.env.obs_height * config.env.obs_width * 3  # flatten成一维向量
config.env.obs_merge_manually = False
config.env.message_distribution_description_dim = 2
if config.env.obs_isimage == False:
    config.env.obs_merge_manually = False
    config.env.messages_shape = [1, config.env.n_agents - 1, config.env.obs_height * config.env.obs_width]
    config.env.message_shape = [1, config.env.obs_height * config.env.obs_width]
else:
    config.env.messages_shape = [1, config.env.n_agents - 1, config.env.obs_height, config.env.obs_width]
    config.env.message_shape = [1, config.env.obs_height, config.env.obs_width]
config.env.dim_action = default_dim_action
config.env.view_size = int((config.env.obs_height - 1) / 2)

# ==================================================

have_sender = False
for i in range(len(config.env.is_sender_list)):
    if config.env.is_sender_list[i]:
        have_sender = True
config.env.have_sender = have_sender
config.env.calculate_part_of_upper_level = have_sender and not config.env.fixed_signaling_scheme

# ==================================================

if wandb_config:
    # config.nn.actor_howmany_linear = wandb_config.actor_howmany_linear
    # config.nn.critic_howmany_linear = wandb_config.critic_howmany_linear
    config.alg.lr_actor = wandb_config.lr_actor
    config.alg.lr_critic = wandb_config.lr_critic
    config.alg.epsilon_start = wandb_config.epsilon_start
    config.alg.epsilon_decay = wandb_config.epsilon_decay
    config.alg.gamma = wandb_config.gamma
    # config.nn.kernel = wandb_config.kernel
    # config.nn.n_filters = wandb_config.n_filters
    # config.nn.n_h2 = wandb_config.n_h2
    config.nn.target_critic_tau = wandb_config.target_critic_tau

    # config.env.obs_vector = config.env.obs_height * config.env.obs_width * config.nn.n_filters  # filters更新了，这个也要更新

    if have_sender:
        config.alg.lr_signal = wandb_config.lr_signal
        config.alg.sender_objective_alpha = wandb_config.sender_objective_alpha
