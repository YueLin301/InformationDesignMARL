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

    torch.autograd.set_detect_anomaly(True)

    i_episode = 0
    buffer = buffer_class()
    while i_episode < config.train.n_episodes:
        run_an_episode(env, sender, receiver, config, device, pls_render=False, buffer=buffer)
        i_episode += 1

        if config.train.batch_size <= buffer.data_size:
            batch = buffer.sample_a_batch(batch_size=config.train.batch_size)

            obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']]
            aj = batch.data[batch.name_dict['aj']]
            pij = batch.data[batch.name_dict['pij']]

            critic = receiver.critic_Qj
            input_critic = obs_and_message_receiver
            a = aj
            pi = pij

            q_table = critic(input_critic)
            q = q_table[range(len(a)), a]
            v = receiver.calculate_v_foractor(critic, input_critic, pi)
            advantage = q - v
            pi_a = pi[range(len(a)), a]
            actor_obj = advantage.detach() * torch.log(pi_a)
            actor_obj_mean = torch.mean(actor_obj)

            net_optimizer = receiver.actor_optimizer
            obj = actor_obj_mean
            net = receiver.actor

            net_optimizer.zero_grad()
            net_grad = torch.autograd.grad(obj, list(net.parameters()), retain_graph=True)
            net_params = list(net.parameters())
            for layer in range(len(net_params)):
                net_params[layer].grad = - net_grad[layer]
                net_params[layer].grad.data.clamp_(-1, 1)
            net_optimizer.step()

            buffer.reset()

        if not i_episode % config.train.n_episodes:
            completion_rate = i_episode / config.train.n_episodes
            print('Task completion:\t{:.1%}'.format(completion_rate))
            sender.save_models()
            receiver.save_models()

    return


if __name__ == '__main__':
    import torch
    from exp_harvest.configs.exp0_test import config

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)
    env, sender, receiver = set_Env_and_Agents(config, device)

    train(env, sender, receiver, config, device)

    print('all done.')
