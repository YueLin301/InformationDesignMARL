from exp_recommendation.rec_utils import print_params, set_seed
from exp_recommendation.train import train, set_Env_and_Agents
from exp_recommendation.configs.exp1_fixed_signaling import config


def main(config):
    for myseed in range(1):
        set_seed(myseed)

        env, pro, hr = set_Env_and_Agents(config)

        print_params(pro, hr)
        fake_buffer = train(config, env, pro, hr)
        print_params(pro, hr)
        plot_all(fake_buffer)

    print('----------------------------------------')
    print('All done.')
    return


if __name__ == '__main__':
    main(config)
