from env import recommendation
from exp_recommendation.agent_class import pro_class, hr_class
from exp_recommendation.episode_generator import run_an_episode

from exp_recommendation.rec_utils import print_params, validate


def set_Env_and_Agents(config):
    print('----------------------------------------')
    print('Setting agents and env.')

    env = recommendation.recommendation_env(config.env)
    pro = pro_class(config,env.rewardmap_HR)
    hr = hr_class(config)

    pro.build_connection(hr)
    hr.build_connection(pro)

    return env, pro, hr


def train(config, env, pro, hr):
    print('----------------------------------------')
    fake_buffer = []  # for ploting curves
    print('Start training.')
    i_episode = 0
    while i_episode < config.train.n_episodes:
        buffer, fake_buffer = run_an_episode(env, pro, hr, fake_buffer)
        i_episode += config.env.sample_n_students

        hr.update_ac(buffer)
        pro.update_c(buffer)

        if not config.pro.fixed_signaling_scheme:
            buffer, fake_buffer = run_an_episode(env, pro, hr, fake_buffer)
            i_episode += config.env.sample_n_students
            pro.update_infor_design(buffer)

        if not i_episode % 1e3:
            completion_rate = i_episode / config.train.n_episodes
            print('Task completion:\t{:.1%}'.format(completion_rate))

    return fake_buffer
