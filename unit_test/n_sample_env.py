from utils.configdict import ConfigDict
from env.recommendation import recommendation_env
import torch

config_env = ConfigDict()
config_env.prob_good = 1 / 3
config_env.num_sample = 1
config_env.reward_magnification_factor = 1
config_env.have_sender = True
config_env.sample_n_students = 100
# config_env.sample_n_students = 1

config_env.rewardmap_professor = [[-1, 10],
                                  [-1, 10]]
config_env.rewardmap_HR = [[-1, -10],
                           [-1, 10]]

if __name__ == '__main__':
    env = recommendation_env(config_env)
    obs = env.reset()

    a = torch.ones_like(obs)
    rewards_pro,rewards_hr = env.step(obs, a)
    print('haha')
