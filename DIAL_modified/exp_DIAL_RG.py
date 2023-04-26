import sys
import os
from pathlib import Path

# sys.path.append('../')

cwd = Path(os.getcwd())
sys.path.append(str(cwd.parent))

# from pprint import pprint
# pprint(sys.path)

# from DIAL_origin.agent import DRU
import torch
from torch.nn.utils import clip_grad_norm_

from exp_reaching_goals.network import critic

from exp_reaching_goals.reaching_goals_utils import plot_with_wandb, init_wandb
from exp_reaching_goals.buffer_class import buffer_class
from exp_reaching_goals.episode_generator import run_an_episode


class DRU_RG():
    def __init__(self, sigma, device=torch.device('cpu')):
        self.sigma = sigma
        self.scale = 2 * 20
        self.device = device

    def regularize(self, m):
        m_reg_raw = m + torch.randn(m.size()).to(self.device) * self.sigma
        # m_reg = torch.softmax(m_reg_raw, 0)
        m_reg = torch.softmax(m_reg_raw, 1)
        return m_reg

    def discretize(self, m):
        return torch.sigmoid((m.gt(0.5).float() - 0.5) * self.scale)

    def forward(self, m, train_mode):
        if train_mode:
            return self.regularize(m)
        else:
            return self.discretize(m)


class sender_DIAL():
    def __init__(self, config, device):
        self.name = name = 'sender'
        self.config = config
        self.device = device
        self.id = 0
        self.epsilon = 0

        self.dim_message = dim_message = config.env.map_height * config.env.map_width

        self.critic = critic(config.n_channels.obs_sender, dim_message, config, belongto=name,
                             name='critic', device=device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.sender.lr_critic)
        self.dru = DRU_RG(sigma=2, device=device)

    def build_connection(self, receiver):
        self.receiver = receiver

    def send_message(self, obs, train_mode=True):
        if train_mode:
            q = self.critic(obs)
            message = self.dru.forward(q, train_mode).view(obs.shape[0], 1,
                                                           self.config.env.map_height, self.config.env.map_width)
            phi = None
        else:
            raise NotImplementedError()
        return message, phi


class receiver_DIAL():
    def __init__(self, config, device):
        self.name = name = 'receiver'
        self.config = config
        self.device = device
        self.dim_action = dim_action = config.env.dim_action
        self.gamma = config.receiver.gamma
        self.id = 1

        self.critic = critic(config.n_channels.obs_and_message_receiver, dim_action, config, belongto=name,
                             name='critic', device=device)
        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.receiver.lr_critic)

        self.target_critic = critic(config.n_channels.obs_and_message_receiver, dim_action, config, belongto=name,
                                    name='target_critic', device=device)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def build_connection(self, sender):
        self.sender = sender

    def choose_action(self, obs_and_message):
        q_table = self.critic(obs_and_message)

        pi_raw = torch.zeros_like(q_table, dtype=torch.double)
        _, max_idx = torch.max(q_table, dim=1)
        pi_raw[torch.arange(q_table.shape[0]), max_idx] = 1

        # epsilon-greedy
        n_action = q_table.shape[1]
        p_eps = self.config.receiver.eps / n_action
        pi = pi_raw * (1 - 2 * p_eps) + p_eps

        a = torch.multinomial(pi, num_samples=1, replacement=True, )[:, 0]
        return a, q_table

    def update_for_all(self, batch):
        rj_raw = batch.data[batch.name_dict['rj']]
        rj = rj_raw[0]

        obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']]
        q_table = self.critic(obs_and_message_receiver)
        aj = batch.data[batch.name_dict['a']]
        q_a = q_table[torch.arange(q_table.shape[0]), aj]

        obs_and_message_receiver_next = batch.data[batch.name_dict['obs_and_message_receiver_next']]
        q_table_next = self.target_critic(obs_and_message_receiver_next)
        aj_next = batch.data[batch.name_dict['a_next']]
        q_prime = q_table_next[torch.arange(q_table_next.shape[0]), aj_next]

        self.critic_optimizer.zero_grad()
        self.sender.critic_optimizer.zero_grad()

        td_loss = self.critic_loss_criterion(rj + self.gamma * q_prime, q_a)
        td_loss.backward()

        clip_grad_norm_(list(self.critic.parameters()), max_norm=1)
        clip_grad_norm_(list(self.sender.critic.parameters()), max_norm=1)

        self.critic_optimizer.step()
        self.sender.critic_optimizer.step()

        tau = self.config.nn.target_critic_tau
        for tar, cur in zip(self.target_critic.parameters(), self.critic.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)


def train(env, sender, receiver, config, device, using_wandb=False, seed=None):
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
        run_an_episode(env, sender, receiver, config, device, pls_render=False, buffer=buffer)
        i_episode += 1
        i_save_flag += 1

        batch = buffer.sample_a_batch(batch_size=buffer.data_size)
        if using_wandb and not (i_episode % record_length):
            plot_with_wandb(chart_name_list, batch, i_episode, sender_honest=config.sender.honest)
        receiver.update_for_all(batch)
        buffer.reset()
        # print(i_episode)
    if using_wandb:
        run_handle.finish()

    return


if __name__ == '__main__':
    # debug_flag = True
    debug_flag = False

    from DIAL_modified.config_RG import config
    from env import reaching_goals
    import wandb
    from exp_reaching_goals.mykey import wandb_login_key
    from exp_reaching_goals.reaching_goals_utils import set_seed

    if debug_flag:
        device_name = 'cpu'
        myseeds = [0]
    else:
        # device_name = 'cuda:0'
        device_name = input("device_name:")
        seeds_raw = input("input seeds:").split(' ')
        myseeds = [int(i) for i in seeds_raw]
        wandb.login(key=wandb_login_key)

    device = torch.device(device_name)

    for seed in myseeds:
        set_seed(seed)

        sender = sender_DIAL(config, device)
        receiver = receiver_DIAL(config, device)
        sender.build_connection(receiver)
        receiver.build_connection(sender)
        env = reaching_goals.reaching_goals_env(config.env)

        if debug_flag:
            train(env, sender, receiver, config, device, using_wandb=False, seed=seed)
        else:
            train(env, sender, receiver, config, device, using_wandb=True, seed=seed)
