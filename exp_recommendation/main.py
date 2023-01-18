import matplotlib.pyplot as plt
from exp_recommendation.rec_utils import print_params, set_seed, plot_create_canvas, plot_all, validate
from exp_recommendation.train import train, set_Env_and_Agents

import wandb
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

# from exp_recommendation.configs.exp6_final import config
# from exp_recommendation.configs.exp6_final_lambda0 import config
# from exp_recommendation.configs.exp6_final_lambda1 import config
# from exp_recommendation.configs.exp6_final_lambda1_5 import config
# from exp_recommendation.configs.exp6_final_lambda2 import config
# from exp_recommendation.configs.exp6_final_lambda2_5 import config
from exp_recommendation.configs.exp6_final_lambda3_5 import config
# from exp_recommendation.configs.exp6_final_lambda5 import config
# from exp_recommendation.configs.exp6_final_lambda100 import config


def main(config, seeds, using_wandb=False, group_name=None):
    if not using_wandb:
        canvas = plot_create_canvas()
    else:
        wandb.login(key=wandb_login_key)

    for myseed in seeds:
        set_seed(myseed)

        env, pro, hr = set_Env_and_Agents(config)

        if not using_wandb:
            print_params(pro, hr)
            validate(pro, hr)

        fake_buffer = train(config, env, pro, hr, using_wandb=using_wandb, group_name=group_name)

        if not using_wandb:
            print_params(pro, hr)
            validate(pro, hr)
            plot_all(fake_buffer, *canvas)
    print('----------------------------------------')
    print('All done.')

    plt.show()


if __name__ == '__main__':
    seeds = [i for i in range(1, 10)]
    main(config, seeds=seeds, using_wandb=True, group_name="recommendation_exp6_lambda=3.5")
