import torch
import os.path
from exp_harvest.configs.exp0_test import config
from agent_class import sender_class, receiver_class
from exp_harvest.harvest_utils import generate_receiver_obs_and_message, obs_list_totorch
from exp_harvest.buffer_class import buffer_class
import torch

from env.harvest import Env

import imageio


def run_an_episode(env, sender, receiver, config, device, pls_render):
    buffer = buffer_class()

    done = False
    observations_np = env.reset()
    observations = obs_list_totorch(observations_np, device)
    step = 0
    if pls_render:
        filename = os.path.join(config.path.saved_episode, str(step))
        env.render(filename)

    while not done:
        message, phi = sender.send_message(observations[sender.id])
        obs_and_message_sender = torch.cat([observations[sender.id], message], dim=1)
        obs_and_message_receiver = generate_receiver_obs_and_message(observations[receiver.id], message)

        a_sender, pi_sender = sender.choose_action(obs_and_message_sender)
        a_receiver, pi_receiver = receiver.choose_action(obs_and_message_receiver)

        observations_next_np, rewards, done, info = env.step([int(a_sender), int(a_receiver)])
        observations_next = obs_list_totorch(observations_next_np, device)
        step += 1
        if pls_render:
            filename = os.path.join(config.path.saved_episode, str(step))
            env.render(filename)

        transition = [observations[sender.id], message, phi, obs_and_message_receiver, a_sender, pi_sender, a_receiver,
                      pi_receiver, rewards[sender.id], rewards[receiver.id]]
        buffer.add(transition)

        observations = observations_next

    if pls_render:
        imgs = []
        for file_idx in range(config.env.max_steps + 1):
            filename = os.path.join(config.path.saved_episode, str(file_idx) + '.png')
            img = imageio.v2.imread(filename)
            imgs.append(img)
        imageio.mimsave(os.path.join(config.path.saved_episode, 'generated_episode.gif'), imgs, duration=0.5)

    return buffer


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    env = Env(config_env=config.env)
    sender = sender_class(config=config, device=device)
    receiver = receiver_class(config=config, device=device)

    # run_an_episode(env, sender, receiver, config, device, pls_render=True)

    import time

    t0 = time.time()
    # for i in range(int(1e3)):
    for i in range(1):
        buffer = run_an_episode(env, sender, receiver, config, device, pls_render=False)
    print(time.time() - t0)

    print('all done.')
