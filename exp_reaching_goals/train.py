import wandb
import math
# from exp_reaching_goals.agent_class import sender_class, receiver_class
from exp_reaching_goals.agent_formal_constrained import sender_class, receiver_class
from exp_reaching_goals.agent_PG_class import sender_PG_class
from exp_reaching_goals.episode_generator import run_an_episode
from exp_reaching_goals.buffer_class import buffer_class
from exp_reaching_goals.reaching_goals_utils import plot_with_wandb, init_wandb
from env.reaching_goals import reaching_goals_env


def set_Env_and_Agents(config, device):
    print('----------------------------------------')
    print('Setting agents and env.')

    env = reaching_goals_env(config.env)
    if not config.sender.honest:
        if hasattr(config.sender, 'type'):
            sender = sender_PG_class(config=config, device=device)
        else:
            sender = sender_class(config=config, device=device)
    else:
        sender = None
    receiver = receiver_class(config=config, device=device)

    if not config.sender.honest:
        sender.build_connection(receiver)
        receiver.build_connection(sender)

    if config.receiver.load:
        receiver.load_models(path=config.receiver.load_path)
        # receiver.load_models()

    return env, sender, receiver


def train(env, sender, receiver, config, device, using_wandb=False, seed=None):
    print('----------------------------------------')
    print('Training.')

    if using_wandb:
        chart_name_list, run_handle = init_wandb(config)
        if not seed is None:
            run_handle.tags = run_handle.tags + (str(seed),)

    i_episode = 0
    i_save_flag = 0
    buffer = buffer_class()
    while i_episode < config.train.n_episodes:
        while buffer.data_size <= buffer.capacity:
            run_an_episode(env, sender, receiver, config, device, pls_render=False, buffer=buffer)
            i_episode += 1
            i_save_flag += 1

        batch = buffer.sample_a_batch(batch_size=buffer.data_size)
        if using_wandb:
            plot_with_wandb(chart_name_list, batch, sender_honest=config.sender.honest)
        update_vars_receiver = receiver.calculate_for_updating(batch)
        receiver.update(*update_vars_receiver)
        buffer.reset()

        if not config.sender.honest and not config.receiver.blind:
            while buffer.data_size <= buffer.capacity:
                run_an_episode(env, sender, receiver, config, device, pls_render=False, buffer=buffer)
                i_episode += 1
            batch = buffer.sample_a_batch(batch_size=buffer.data_size)
            if using_wandb:
                plot_with_wandb(chart_name_list, batch, sender_honest=config.sender.honest)
            update_vars_sender = sender.calculate_for_updating(batch)
            sender.update(*update_vars_sender)
            buffer.reset()

        if i_save_flag >= config.train.period:
            completion_rate = i_episode / config.train.n_episodes
            print('Task completion:\t{:.1%}'.format(completion_rate))
            if not config.sender.honest:
                sender.save_models()
            receiver.save_models()
            i_save_flag -= config.train.period

    if using_wandb:
        run_handle.finish()

    return


if __name__ == '__main__':
    import torch
    from exp_harvest.configs.exp0_test import config

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(device)
    env, sender, receiver = set_Env_and_Agents(config, device)

    train(env, sender, receiver, config, device)

    print('all done.')
