import sys

sys.path.append('../')

import torch
from exp_reaching_goals.train import train, set_Env_and_Agents
from exp_reaching_goals.reaching_goals_utils import set_seed

import wandb
from exp_reaching_goals.mykey import wandb_login_key


def main(config, seeds, device_name, using_wandb=False):
    device = torch.device(device_name)
    print(device)

    if using_wandb:
        wandb.login(key=wandb_login_key)

    for myseed in seeds:
        set_seed(myseed)
        env, sender, receiver = set_Env_and_Agents(config, device)
        train(env, sender, receiver, config, device, using_wandb=using_wandb, seed=myseed)

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

    '''tests in previous version'''
    # from exp_reaching_goals.configs.exp3a_aligned_map3 import config
    # from exp_reaching_goals.configs.exp3b_aligned_map5_2 import config
    # from exp_reaching_goals.configs.exp8a_baseline_blind import config
    # from exp_reaching_goals.configs.exp8b_baseline_blind import config
    # from exp_reaching_goals.configs_formal.RG_map3_no_punish import config
    # from exp_reaching_goals.configs_formal.RG_map3_no_punish_SG import config

    ''' comparisons: PG, PGOC, SG, SGOC '''
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0_PG import config # PG
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0005_PG import config  # PGOC
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0 import config # SG
    from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0005_eps0 import config  # SGOC

    '''control oj'''
    # from exp_reaching_goals.configs_oj.RG_map3_SGOC_oj_00 import config  # message
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0005_eps0 import config  # j pos, message
    # from exp_reaching_goals.configs_oj.RG_map3_SGOC_oj_11 import config  # j pos, j apple pos, message

    '''5x5 map'''
    # from exp_reaching_goals.configs_bigger_map.RG_map5_PGOC import config  # PGOC
    # from exp_reaching_goals.configs_bigger_map.RG_map5_SGOC import config  # SGOC

    main(config, seeds, device_name, using_wandb=using_wandb)
