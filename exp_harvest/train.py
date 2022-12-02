from env.harvest import Env
from exp_harvest.agent_class import sender_class, receiver_class
from exp_harvest.episode_generator import run_an_episode


def set_Env_and_Agents(config, device):
    print('----------------------------------------')
    print('Setting agents and env.')

    env = Env(config_env=config.env)
    sender = sender_class(config=config, device=device)
    receiver = receiver_class(config=config, device=device)

    sender.build_connection(receiver)
    receiver.build_connection(sender)

    return env, sender, receiver


def train(env, sender, receiver, config, device):
    print('----------------------------------------')
    print('Training.')
    i_episode = 0
    while i_episode < config.train.n_episodes:
        buffer = run_an_episode(env, sender, receiver, config, device, pls_render=False)
        i_episode += 1

        sender.update_ac(buffer)
        receiver.update_ac(buffer)

        buffer = run_an_episode(env, sender, receiver, config, device, pls_render=False)
        i_episode += 1
        sender.update_infor_design(buffer)

        if not i_episode % config.train.n_episodes:
            completion_rate = i_episode / config.train.n_episodes
            print('Task completion:\t{:.1%}'.format(completion_rate))
            sender.save_models()
            receiver.save_models()

    return
