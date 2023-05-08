import wandb
import math
from exp_morej_RG.agent_formal_constrained import sender_class, receiver_class
# from exp_morej_RG.agent_PG_class import sender_PG_class
from exp_morej_RG.episode_generator import run_an_episode
from exp_morej_RG.buffer_class import buffer_class
from exp_morej_RG.reaching_goals_utils import plot_with_wandb, init_wandb
from env.reaching_goals_v2 import reaching_goals_env


def set_Env_and_Agents(config, device):
    print('----------------------------------------')
    print('Setting agents and env.')

    env = reaching_goals_env(config.env, device)
    if hasattr(config.sender, 'type'):
        raise NotImplementedError()
        # sender = sender_PG_class(config=config, device=device)
    else:
        sender = sender_class(config=config, device=device)

    receiver_list = []
    for i in range(config.env.nj):
        receiver_list.append(receiver_class(config=config, device=device, id=i + 1))

    sender.build_connection(receiver_list)
    for receiver in receiver_list:
        receiver.build_connection(sender)

    return env, sender, receiver_list


def train(env, sender, receiver_list, config, device, using_wandb=False, seed=None):
    print('----------------------------------------')
    print('Training.')

    if using_wandb:
        chart_name_list, run_handle = init_wandb(config)
        if not seed is None:
            run_handle.tags = run_handle.tags + (str(seed),)

    record_length = 100
    i_episode = 0
    i_save_flag = 0
    buffer = buffer_class()
    while i_episode < config.train.n_episodes:
        while buffer.data_size <= buffer.capacity:
            run_an_episode(env, sender, receiver_list, config, pls_render=False, buffer=buffer)
            i_episode += 1
            i_save_flag += 1

        batch = buffer.sample_a_batch(batch_size=buffer.data_size)
        if using_wandb and not (i_episode % record_length):
            plot_with_wandb(chart_name_list, batch, i_episode)
        for receiver in receiver_list:
            update_vars_receiver = receiver.calculate_for_updating(batch)
            receiver.update(*update_vars_receiver)
        buffer.reset()

        while buffer.data_size <= buffer.capacity:
            run_an_episode(env, sender, receiver_list, config, pls_render=False, buffer=buffer)
            i_episode += 1
        batch = buffer.sample_a_batch(batch_size=buffer.data_size)
        if using_wandb and not (i_episode % record_length):
            plot_with_wandb(chart_name_list, batch, i_episode)
        update_vars_sender = sender.calculate_for_updating(batch)
        sender.update(*update_vars_sender)
        buffer.reset()

        if i_save_flag >= config.train.period:
            completion_rate = i_episode / config.train.n_episodes
            print('Task completion:\t{:.1%}'.format(completion_rate))
            sender.save_models()
            for receiver in receiver_list:
                receiver.save_models()
            i_save_flag -= config.train.period

    if using_wandb:
        run_handle.finish()

    return
