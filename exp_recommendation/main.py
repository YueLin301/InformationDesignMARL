import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import torch

from exp_recommendation.rec_utils import print_params, set_seed, plot_create_canvas, plot_all, validate
from exp_recommendation.train import train, set_Env_and_Agents

import wandb
import time
from exp_recommendation.mykey import wandb_login_key


# from exp_recommendation.configs.exp1a_fixed_signaling_scheme1 import config
# from exp_recommendation.configs.exp1b_fixed_signaling_scheme2 import config
# from exp_recommendation.configs.exp1c_fixed_signaling_scheme3 import config
# from exp_recommendation.configs.exp1d_fixed_signaling_scheme4 import config

# from exp_recommendation.configs.exp2_fixed_receiver_policy import config

# from exp_recommendation.configs.exp3_init_scheme3_no_constraint  import config

# from exp_recommendation.configs.exp4a_init_scheme3_equilibrium3rd  import config
# from exp_recommendation.configs.exp4b_init_scheme3_equilibrium4th  import config

# from exp_recommendation.configs.exp5_init_receiver_lambda2 import config
# from exp_recommendation.configs.exp5a_init_receiver_lambda0 import config
# from exp_recommendation.configs.exp5c_init_receiver_lambda5 import config
# from exp_recommendation.configs.exp5d_init_receiver_lambda10 import config


def main(config, seeds, using_wandb=False, group_name=None, pro_type='regularized', device_name='cpu'):
    assert pro_type in ['baseline', 'formal_constrained', 'regularized']
    baseline = True if pro_type == 'baseline' else False
    device = torch.device(device_name)

    if not using_wandb:
        canvas = plot_create_canvas()
    else:
        wandb.login(key=wandb_login_key)

    for myseed in seeds:
        set_seed(myseed)

        env, pro, hr = set_Env_and_Agents(config, pro_type, device=device)

        if not using_wandb and not baseline:
            print_params(pro, hr)
            validate(pro, hr, device)

        fake_buffer = train(config, env, pro, hr, using_wandb=using_wandb, group_name=group_name, seed=myseed,
                            pro_type=pro_type, device=device)

        if not using_wandb and not baseline:
            print_params(pro, hr)
            validate(pro, hr, device)
        if not using_wandb:
            plot_all(fake_buffer, *canvas)
        else:
            time.sleep(120)
    print('----------------------------------------')

    plt.show()


if __name__ == '__main__':
    # device_name = input("device_name:")
    device_name = 'cuda:0'

    # seeds = [i for i in range(0, 10)]
    seeds = [0]

    ############################################################

    from exp_recommendation.configs.exp6_final_baseline_0 import config

    # from exp_recommendation.configs.exp6_final_baseline_5 import config

    main(config, seeds=seeds, pro_type='baseline', device_name=device_name)
    # main(config, seeds=seeds, pro_type='baseline', using_wandb=True,
    #      group_name="recom_baseline_" + str(config.pro.sender_objective_alpha), device_name=device_name)

    ############################################################

    # from exp_recommendation.configs.exp6_final_lambda0 import config
    # from exp_recommendation.configs.exp6_final_lambda2_25 import config

    # main(config, seeds=seeds, pro_type='regularized', device_name=device_name)
    # main(config, seeds=seeds, pro_type='regularized', using_wandb=True, group_name="recommendation_exp6_lambda="+str(config.pro.sender_objective_alpha), device_name=device_name)

    ############################################################

    # from exp_recommendation.formal_config import config

    # main(config, seeds=seeds, pro_type='formal_constrained', device_name=device_name)
    # main(config, seeds=seeds, pro_type='formal_constrained', using_wandb=True,
    #      group_name="recom_lam=" + str(config.pro.sender_objective_alpha) + "_eps=" + str(config.pro.constraint_right), device_name=device_name)
    # recom_lam=5_eps=0.3

    ############################################################

    # from exp_recommendation.formal_config import config
    #
    # # lam_list = [2.5 * i for i in range(0, 5)]
    # # lam_list = [2.5 * i + 1.25 for i in range(0, 4)]
    # # lam_list = [1.25 * i for i in range(0, 9)]
    # lam_list = [1.25 * i for i in range(0, 2)]
    # eps_list = [0.15 * i for i in range(0, 5)]
    #
    # last_finished_ij = [-1, -1]  # epsilon, lambda; for restart
    #
    # for i in range(len(eps_list)):
    #     for j in range(len(lam_list)):
    #         if i <= last_finished_ij[0] and j <= last_finished_ij[1]:
    #             continue
    #         config.pro.constraint_right = eps_list[i]
    #         config.pro.sender_objective_alpha = lam_list[j]
    #         main(config, seeds=seeds, pro_type='formal_constrained', using_wandb=True,
    #              group_name="recom_lam=" + str(config.pro.sender_objective_alpha) + "_eps=" + str(
    #                  config.pro.constraint_right))
    #
    # print('----------------------------------------')
    # print('All done.')
