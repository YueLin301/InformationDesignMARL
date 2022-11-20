import matplotlib.pyplot as plt
from exp_recommendation.rec_utils import print_params, set_seed, plot_create_canvas, plot_all, validate
from exp_recommendation.train import train, set_Env_and_Agents
from exp_recommendation.configs.exp1a_fixed_signaling_scheme1 import config
# from exp_recommendation.configs.exp1b_fixed_signaling_scheme2 import config


def main(config):
    canvas = plot_create_canvas()
    for myseed in range(10):
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
    main(config)
