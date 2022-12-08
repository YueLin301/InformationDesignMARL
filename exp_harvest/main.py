import torch
from exp_harvest.train import train, set_Env_and_Agents
from exp_harvest.harvest_utils import set_seed

from exp_harvest.configs.exp1_7 import config


def main(config, myseed=0):
    set_seed(myseed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)

    env, sender, receiver = set_Env_and_Agents(config, device)

    train(env, sender, receiver, config, device)

    print('----------------------------------------')
    print('All done.')


if __name__ == '__main__':
    main(config, myseed=0)
