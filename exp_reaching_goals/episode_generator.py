import torch
import os.path
from agent_class import sender_class, receiver_class
# from exp_reaching_goals.reaching_goals_utils import obs_list_totorch
from exp_reaching_goals.buffer_class import buffer_class
from exp_reaching_goals.configs.exp1_aligned import config

from env.reaching_goals import reaching_goals_env

import imageio


def run_an_episode(env, sender, receiver, config, device, pls_render, buffer):
    done = False
    obs_sender = env.reset().to(device)

    step = 0
    if pls_render:
        filename = os.path.join(config.path.saved_episode, str(step))
        env.render(filename=filename)
    while not done:

        message, phi = sender.send_message(obs_sender)
        if pls_render:
            filename = os.path.join(config.path.saved_episode, str(step))
            message_pos = torch.nonzero(message == 1).squeeze(dim=0)[2:]
            env.render(message_pos, filename)
        a_receiver, pi = receiver.choose_action(message)

        obs_sender_next, rewards, done = env.step(int(a_receiver))

        half_transition = [obs_sender.to(device), message, phi, a_receiver, pi,
                           torch.tensor(rewards[sender.id], dtype=torch.double).to(device).unsqueeze(dim=0),
                           torch.tensor(rewards[receiver.id], dtype=torch.double).to(device).unsqueeze(dim=0)]
        half_transition_clone = [half_transition[i].clone() for i in range(len(half_transition))]

        if not done:  # the last transition
            buffer.add_half_transition(half_transition, '1st')
        if step:  # the first transition
            buffer.add_half_transition(half_transition_clone, '2nd')

        step += 1
        if pls_render:
            filename = os.path.join(config.path.saved_episode, str(step))
            message_pos = torch.nonzero(message == 1).squeeze(dim=0)[2:]
            env.render(message_pos, filename)
        obs_sender = obs_sender_next.to(device)

    if pls_render:
        imgs = []
        for file_idx in range(config.env.max_step + 1):
            filename = os.path.join(config.path.saved_episode, str(file_idx) + '.png')
            # img = imageio.v2.imread(filename)
            img = imageio.imread(filename)
            imgs.append(img)
        imageio.mimsave(os.path.join(config.path.saved_episode, 'generated_episode.gif'), imgs, duration=0.5)

    return


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    env = reaching_goals_env(config.env)
    sender = sender_class(config=config, device=device)
    receiver = receiver_class(config=config, device=device)

    # run_an_episode(env, sender, receiver, config, device, pls_render=True)

    # import time
    # t0 = time.time()
    # for i in range(int(1e3)):
    #     buffer = run_an_episode(env, sender, receiver, config, device, pls_render=False)
    # print(time.time() - t0)

    buffer = buffer_class()
    run_an_episode(env, sender, receiver, config, device, pls_render=False, buffer=buffer)

    assert torch.cat(buffer.data[buffer.name_dict['obs_sender']][1:]).equal(
        torch.cat(buffer.data[buffer.name_dict['obs_sender_next']][:-1]))
    assert buffer.data[buffer.name_dict['message']][1:] == buffer.data[buffer.name_dict['message_next']][:-1]
    assert buffer.data[buffer.name_dict['obs_and_message_receiver']][1:] == buffer.data[buffer.name_dict[
        'obs_and_message_receiver_next']][:-1]
    ri = buffer.data[buffer.name_dict['ri']]
    batch = buffer.sample_a_batch(33)
    ri2 = batch.data[batch.name_dict['ri']]

    phi = buffer.data[buffer.name_dict['phi']]

    temp_grad = torch.autograd.grad(torch.sum(phi[0]), list(sender.signaling_net.parameters()), retain_graph=True)

    print('all done.')
