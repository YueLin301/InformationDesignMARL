from env import recommendation
from exp_recommendation.agent_class import pro_class, hr_class
from exp_recommendation.episode_generator import run_an_episode


def set_Env_and_Agents(config):
    print('----------------------------------------')
    print('Setting agents and env.')

    env = recommendation.recommendation_env(config.env)
    pro = pro_class(config)
    hr = hr_class(config)

    pro.build_connection(hr)
    hr.build_connection(pro)

    return env, pro, hr


def train(config, env, pro, hr):
    print('----------------------------------------')
    fake_buffer = []
    print('Start training.')
    i_episode = 0
    while i_episode < config.train.n_episodes:
        buffer = []
        for _ in range(config.train.howoften_update):
            transition = run_an_episode(env, pro, hr, ith_episode=i_episode, pls_print=False)
            fake_buffer.append(transition.transition)
            buffer.append(transition)
            i_episode += 1
        hr.update_ac(buffer)
        pro.update_c(buffer)

        buffer = []
        if not config.pro.fixed_signaling_scheme:
            for _ in range(config.train.howoften_update):
                transition = run_an_episode(env, pro, hr, ith_episode=i_episode, pls_print=False)
                fake_buffer.append(transition.transition)
                buffer.append(transition)
                i_episode += 1
            pro.update_infor_design(buffer)

        if not i_episode % 1e3:
            completion_rate = i_episode / config.train.n_episodes
            print('Task completion:\t{:.1%}'.format(completion_rate))

    return fake_buffer
