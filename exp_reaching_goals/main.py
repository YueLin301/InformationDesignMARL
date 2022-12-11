import torch
from exp_reaching_goals.train import train, set_Env_and_Agents
from exp_reaching_goals.reaching_goals_utils import set_seed

from exp_reaching_goals.configs.exp3_aligned import config


def main(config, myseed=0):
    set_seed(myseed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)

    env, sender, receiver = set_Env_and_Agents(config, device)

    # train(env, sender, receiver, config, device, using_wandb=True)
    train(env, sender, receiver, config, device, using_wandb=False)

    print('----------------------------------------')
    print('All done.')


if __name__ == '__main__':
    import time

    t0 = time.time()
    main(config, myseed=0)
    print('time:', time.time() - t0)
