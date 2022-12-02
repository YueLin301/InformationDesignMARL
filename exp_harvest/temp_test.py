import glob
import os.path
from exp_harvest.configs.exp1_harvest import config
from agent_class import sender_class, receiver_class
from exp_harvest.harvest_utils import generate_receiver_obs_and_message, obs_list_totorch
import torch

from env.harvest import Env

import matplotlib.pyplot as plt


def unittest():
    return


def generate_animation_failed():
    import imageio

    directory_name = './generated_episode'
    all_name = os.path.join(directory_name, '*.png')

    imgs = []  # for storing the generated generated_episode
    for filename in glob.glob(all_name):
        # img = mpimg.imread(filename)
        img = imageio.imread(filename)
        imgs.append(img)
    imageio.mimsave('./generated_episode/generated_episode.gif', imgs)


def generate_animation():
    import imageio

    directory_name = './generated_episode'

    imgs = []
    for file_idx in range(51):
        filename = os.path.join(directory_name, str(file_idx) + '.png')
        img = imageio.imread(filename)
        imgs.append(img)
    imageio.mimsave('./generated_episode/generated_episode.gif', imgs, duration=0.5)


def unittest_env():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = Env(config_env=config.env)
    sender = sender_class(config=config, device=device)
    receiver = receiver_class(config=config, device=device)

    directory_name = './generated_episode'

    done = False
    observations_np = env.reset()
    observations = obs_list_totorch(observations_np, device)
    step = 0
    filename = os.path.join(directory_name, str(step))
    env.render(filename)

    while not done:
        message, phi = sender.send_message(observations[sender.id])
        obs_and_message_receiver = generate_receiver_obs_and_message(observations[receiver.id], message)

        a_sender, pi_sender, logpi_sender = sender.choose_action(observations[sender.id])
        a_receiver, pi_receiver, logpi_receiver = receiver.choose_action(obs_and_message_receiver)

        observations_next_np, rewards, done, info = env.step([int(a_sender), int(a_receiver)])
        observations_next = obs_list_totorch(observations_next_np, device)
        step += 1
        filename = os.path.join(directory_name, str(step))
        env.render(filename)
        # buffer
        observations = observations_next

    return


if __name__ == '__main__':
    # unittest()
    # unittest_env()
    generate_animation()
    print('all done.')
