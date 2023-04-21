import sys

sys.path.append('../')

import torch
from exp_reaching_goals.train import train, set_Env_and_Agents
from exp_reaching_goals.reaching_goals_utils import set_seed

import wandb
from exp_reaching_goals.mykey import wandb_login_key


def main(config, seeds, device_name):
    device = torch.device(device_name)
    print(device)

    wandb.login(key=wandb_login_key)

    for myseed in seeds:
        set_seed(myseed)
        env, sender, receiver = set_Env_and_Agents(config, device)
        train(env, sender, receiver, config, device, using_wandb=True, seed=myseed)
        # train(env, sender, receiver, config, device, using_wandb=False, seed=myseed)

    print('----------------------------------------')
    print('All done.')


if __name__ == '__main__':
    device_name = input("device_name:")
    # device_name = 'cpu'

    # seeds = [i for i in range(0, 1)]
    seeds_raw = input("input seeds:").split(' ')
    seeds = [int(i) for i in seeds_raw]

    # from exp_reaching_goals.configs_formal.RG_map3_no_punish import config
    # from exp_reaching_goals.configs_formal.RG_map3_no_punish_SG import config

    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0_noinfo import config # noinfo
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0_PG import config # PG
    from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0005_PG import config  # PGOC
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0 import config # SG
    # from exp_reaching_goals.configs_formal.RG_map3_gam01_lam0005_eps0 import config  # SGOC

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

    main(config, seeds, device_name)
