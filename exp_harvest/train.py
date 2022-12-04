from env.harvest import Env
from exp_harvest.agent_class import sender_class, receiver_class
from exp_harvest.episode_generator import run_an_episode
from exp_harvest.buffer_class import buffer_class


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
    buffer = buffer_class()
    while i_episode < config.train.n_episodes:
        run_an_episode(env, sender, receiver, config, device, pls_render=False, buffer=buffer)
        i_episode += 1

        batch = buffer.sample_a_batch(batch_size=config.train.buffer_size)
        sender.update_ac(batch)
        receiver.update_ac(batch)
        sender.update_infor_design(batch)

        if not i_episode % config.train.n_episodes:
            completion_rate = i_episode / config.train.n_episodes
            print('Task completion:\t{:.1%}'.format(completion_rate))
            sender.save_models()
            receiver.save_models()

    return
