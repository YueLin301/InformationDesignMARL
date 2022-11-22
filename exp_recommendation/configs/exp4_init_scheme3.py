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
config.pro.fixed_signaling_scheme = False

config.pro.lr_pro_critic = 1.5e-2
config.pro.lr_signal = 1.5e-2

config.pro.constraint_right = 0
config.pro.sender_objective_alpha = 0.28125  # Lagrangian lambda
# [0, 0.5] -> [0.25, 0.5] -> [0.25, 0.375] -> [0.25, 0.3125] -> [0.28125, 0.3125]
# -> [0.28125, 0.296875]

config.pro.initialize = True
if config.pro.initialize:
    # signaling scheme of pro is initialized to be optimal
    # config.pro.critic_pro_params = [torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.double),
    #                                 torch.tensor([[1, 1]], dtype=torch.double)]
    config.pro.signaling_params = torch.tensor([[10, 0],
                                                [10, 10]], dtype=torch.double).unsqueeze(dim=0)

# ========================================

'''hr'''
config.hr = ConfigDict()

config.hr.fixed_policy = False

config.hr.lr_critic = 2e-2
config.hr.lr_actor = 2e-2

config.hr.epsilon_start = 0  # epsilon_greedy
config.hr.epsilon_decay = 0  # after running an episode: epsilon <- epsilon * epsilon_decay

config.hr.initialize = True
if config.hr.initialize:
    # config.hr.critic_params = [torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.double),
    #                            torch.tensor([[1, 1]], dtype=torch.double)]

    # policy of hr is initialized to be obedient
    config.hr.actor_params = torch.tensor([[5, -5],
                                           [-5, 5]], dtype=torch.double).unsqueeze(dim=0)

# ========================================
'''train'''
config.train = ConfigDict()

config.train.n_episodes = 3e4
# config.train.n_episodes = 6e4
# config.train.n_episodes = 5e4
# config.train.n_episodes = 4e5
# config.train.n_episodes = 1e6

config.train.howoften_update = 20

config.train.GAE_term = 'TD-error'
