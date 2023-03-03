import torch
from exp_reaching_goals.train import train, set_Env_and_Agents
from exp_reaching_goals.reaching_goals_utils import set_seed

import wandb
from exp_reaching_goals.mykey import wandb_login_key


def main(config, seeds):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(device)

    wandb.login(key=wandb_login_key)

    for myseed in seeds:
        set_seed(myseed)
        env, sender, receiver = set_Env_and_Agents(config, device)
        train(env, sender, receiver, config, device, using_wandb=True)

    print('----------------------------------------')
    print('All done.')


if __name__ == '__main__':
    # seeds = [i for i in range(0, 1)]
    seeds_raw = input("input seeds:").split(' ')
    seeds = [int(i) for i in seeds_raw]

    # config_id = input("which config:")
    # if config_id == '0':
    #     from exp_reaching_goals.configs.exp8b_load_map5_0 import config
    # elif config_id == '0005':
    #     from exp_reaching_goals.configs.exp8b_load_map5_0005 import config
    # else:
    #     raise IOError

    # from exp_reaching_goals.configs.exp8a_baseline_blind import config
    # from exp_reaching_goals.configs.exp8b_baseline_blind import config

    # from exp_reaching_goals.configs.exp3a_aligned_map3 import config
    from exp_reaching_goals.configs.exp3b_aligned_map5_2 import config

    main(config, seeds)
