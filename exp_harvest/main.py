import matplotlib.pyplot as plt
from exp_harvest.harvest_utils import print_params, set_seed, plot_create_canvas, plot_all, validate
from exp_harvest.train import train, set_Env_and_Agents

from exp_harvest.configs.exp1_harvest import config


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
