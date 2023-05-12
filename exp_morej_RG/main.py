import sys

sys.path.append('../')

import torch
from exp_morej_RG.train import train, set_Env_and_Agents
from exp_morej_RG.reaching_goals_utils import set_seed

import wandb
from exp_morej_RG.mykey import wandb_login_key


def main(config, seeds, device_name, using_wandb=False):
    device = torch.device(device_name)
    print(device)

    if using_wandb:
        wandb.login(key=wandb_login_key)

    for myseed in seeds:
        set_seed(myseed)
        env, sender, receiver_list = set_Env_and_Agents(config, device)
        train(env, sender, receiver_list, config, device, using_wandb=using_wandb, seed=myseed)

    print('----------------------------------------')
    print('All done.')


if __name__ == '__main__':
    debug_flag = True
    # debug_flag = False

    if debug_flag:
        device_name = 'cpu'
        # seeds = [i for i in range(0, 1)]
        seeds = [0]
        using_wandb = False
    else:
        device_name = input("device_name:")
        seeds_raw = input("input seeds:").split(' ')
        seeds = [int(i) for i in seeds_raw]
        using_wandb = True

    device_name = 'cuda:0'

    from exp_morej_RG.configs.RG_morej_map3_SGOC import config  # SGOC

    main(config, seeds, device_name, using_wandb=using_wandb)
