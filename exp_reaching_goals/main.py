import sys

sys.path.append('../')

import torch
from exp_reaching_goals.train import train, set_Env_and_Agents
from exp_reaching_goals.reaching_goals_utils import set_seed

import wandb
from exp_reaching_goals.mykey import wandb_login_key


def main(config, seeds, device_name, using_wandb):
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
    device_name = "cpu"
    # device_name = input("device_name:")

    seeds = [0]
    # seeds_raw = input("input seeds:").split(' ')
    # seeds = [int(i) for i in seeds_raw]

    # from exp_reaching_goals.configs_formal.RG_map3_no_punish import config
    # from exp_reaching_goals.configs_formal.RG_map3_no_punish_SG import config

    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0 import config
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0_PG import config
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0_noinfo import config

    # from exp_reaching_goals.configs_formal.RG_map5_no_punish_test1 import config
    # from exp_reaching_goals.configs_formal.RG_map5_no_punish_test2 import config
    # from exp_reaching_goals.configs_formal.RG_map5_no_punish_test3 import config

    # config_id = input("which config:")
    # if config_id == '0':
    #     # from exp_reaching_goals.configs.exp8b_load_map5_0 import config
    #     from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0_eps0 import config
    # elif config_id == '0005':
    #     from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0005_eps0 import config
    # else:
    #     raise IOError

    # from exp_reaching_goals.configs.exp8a_baseline_blind import config
    # from exp_reaching_goals.configs.exp8b_baseline_blind import config

    # from exp_reaching_goals.configs.exp3a_aligned_map3 import config
    # from exp_reaching_goals.configs.exp3b_aligned_map5_2 import config

    from exp_reaching_goals.configs_formal.RG_map3_DGD import config

    main(config, seeds, device_name, using_wandb=False)
