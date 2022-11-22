from utils.configdict import ConfigDict

config_env = ConfigDict()
config_env.prob_good = 1 / 3
config_env.num_sample = 1
config_env.reward_magnification_factor = 1
config_env.have_sender = True


config_env.rewardmap_professor = [[-1, 10],
                                  [-1, 10]]
config_env.rewardmap_HR = [[-1, -10],
                           [-1, 10]]



# config_env.rewardmap_professor = [[0, 1],
#                                   [0, 1]]
# config_env.rewardmap_HR = [[0, -1],
#                            [0, 1]]
