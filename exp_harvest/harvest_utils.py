import torch
import numpy as np


def generate_receiver_obs_and_message(obs_receiver, message):
    # sender position, receiver position, apple position
    # receiver position, message(apple position)

    obs_receiver_position = obs_receiver[:, 1, :, :].unsqueeze(dim=1)
    obs_and_message_receiver = torch.cat([obs_receiver_position, message], dim=1)

    return obs_and_message_receiver


def generate_receiver_obs_and_message_counterfactual(obs_and_message_receiver, counterfactual_message):
    # receiver position, message(apple position)
    # receiver position, counterfactual message(apple position)

    obs_receiver_position = obs_and_message_receiver[:, 0, :, :].unsqueeze(dim=1)
    obs_and_message_receiver = torch.cat([obs_receiver_position, counterfactual_message], dim=1)

    return obs_and_message_receiver


def flatten_layers(gradtensor, dim=0):
    gradtensor_flatten = torch.flatten(gradtensor[0])
    for layerl in range(1, len(gradtensor)):
        temp = torch.flatten(gradtensor[layerl])
        gradtensor_flatten = torch.cat([gradtensor_flatten, temp])
    gradtensor_flatten = gradtensor_flatten.unsqueeze(dim=dim)

    return gradtensor_flatten


def obs_list_totorch(obs_listofnp, device):
    # obs是5*5*3的，[obs1,obs2]，obs1是nparray，要换成[1,3,5,5]的tensor
    obs_listoftorch = []
    for agent_id in range(len(obs_listofnp)):
        obs = obs_listofnp[agent_id].transpose(2, 0, 1)  # 维度2变到维度0的位置，维度0变到维度1的位置，维度1变到维度2的位置,[3,5,5]
        obs = np.array([obs])  # [1,3,5,5]
        # obs = np.array([obs],dtype='float64')  # [1,3,5,5]
        obs = torch.tensor(obs, dtype=torch.double)  # tensor [1,3,5,5]
        obs_listoftorch.append(obs.to(device))
    return obs_listoftorch
