import matplotlib.pyplot as plt
from exp_recommendation.rec_utils import print_params, set_seed, plot_create_canvas, plot_all, validate
from exp_recommendation.train import train, set_Env_and_Agents

# from exp_recommendation.configs.exp1a_fixed_signaling_scheme1 import config
# from exp_recommendation.configs.exp1b_fixed_signaling_scheme2 import config
# from exp_recommendation.configs.exp1c_fixed_signaling_scheme3 import config

# from exp_recommendation.configs.exp2_fixed_receiver_policy import config

# from exp_recommendation.configs.exp3_init_scheme3_no_constraint  import config

# from exp_recommendation.configs.exp4a_init_scheme3_equilibrium3rd  import config
# from exp_recommendation.configs.exp4b_init_scheme3_equilibrium4th  import config

from exp_recommendation.configs.exp5_init_receiver_lambda2 import config
# from exp_recommendation.configs.exp5a_init_receiver_lambda0 import config
# from exp_recommendation.configs.exp5c_init_receiver_lambda5 import config
# from exp_recommendation.configs.exp5d_init_receiver_lambda10 import config

# from exp_recommendation.configs.exp6_final import config
# from exp_recommendation.configs.exp6_final_lambda0 import config
# from exp_recommendation.configs.exp6_final_lambda1 import config
# from exp_recommendation.configs.exp6_final_lambda2 import config
# from exp_recommendation.configs.exp6_final_lambda5 import config
# from exp_recommendation.configs.exp6_final_lambda100 import config


def main(config, seeds_num=10):
    canvas = plot_create_canvas()
    for myseed in range(seeds_num):
        set_seed(myseed)

        env, pro, hr = set_Env_and_Agents(config)

        print_params(pro, hr)
        validate(pro, hr)

        fake_buffer = train(config, env, pro, hr)

        print_params(pro, hr)
        validate(pro, hr)
        plot_all(fake_buffer, *canvas)
    print('----------------------------------------')
    print('All done.')

    plt.show()


if __name__ == '__main__':
    main(config, seeds_num=1)
    # main(config, seeds_num=5)
    # main(config, seeds_num=10)
