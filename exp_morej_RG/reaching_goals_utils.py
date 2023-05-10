import torch
import numpy as np
import wandb
from exp_morej_RG.mykey import wandb_login_key, wandb_project_name, wandb_entity_name


def init_wandb(config):
    # wandb.login(key=wandb_login_key)
    # wandb.init(project=wandb_project_name, entity=wandb_entity_name)
    run_handle = wandb.init(project=wandb_project_name, entity=wandb_entity_name, reinit=True,
                            group=config.main.exp_name,
                            config={"sender_objective_alpha": config.sender.sender_objective_alpha,
                                    "map": config.env.map_height,
                                    "receiver_gamma": config.receiver.gamma})

    chart_name_list = ['reward_sender',
                       'reward_receiver_mean',
                       'social_welfare', ]

    for i in range(config.env.nj):
        chart_name_list.append('reward_receiver_' + str(i + 1))

    for chart_name in chart_name_list:
        wandb.define_metric(chart_name)

    return chart_name_list, run_handle


def plot_with_wandb(chart_name_list, batch, i_episode):
    entry = dict(zip(chart_name_list, [0] * len(chart_name_list)))

    ri = batch.data[batch.name_dict['ri']]
    ri_sum = float(torch.sum(ri))
    entry['reward_sender'] = ri_sum

    rj = batch.data[batch.name_dict['rj']]
    rj_mean = float(torch.sum(torch.mean(rj, dim=1)))
    entry['reward_receiver_mean'] = rj_mean

    r_tot = ri + torch.sum(rj, dim=1).squeeze()
    r_tot_sum = float(torch.sum(r_tot))
    entry['social_welfare'] = r_tot_sum

    rj_each = torch.sum(rj, dim=0)
    for i in range(3, len(chart_name_list)):
        entry[chart_name_list[i]] = float(rj_each[i - 3])
    wandb.log(entry, step=i_episode // 2)

    return


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


def set_seed(myseed):
    np.random.seed(myseed)  # numpy seed
    torch.manual_seed(myseed)  # torch seed
    torch.cuda.manual_seed(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
