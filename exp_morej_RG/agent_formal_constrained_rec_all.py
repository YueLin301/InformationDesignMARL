import torch
from reaching_goals_utils import flatten_layers
from network import actor, critic_embedding, critic, signaling_net

import numpy as np


def calculate_critic_loss(critic, target_critic, loss_criterion, r, input, a, input_next, a_next, gamma):
    output_table = critic(input)
    target_output_table = target_critic(input_next)
    output = output_table[range(len(a)), a]
    target_output = target_output_table[range(len(a)), a_next]

    td_target = r + gamma * target_output
    critic_loss = loss_criterion(td_target, output)

    return critic_loss


def sender_calculate_critic_loss(critic, target_critic, loss_criterion, r, input, aj, input_next, aj_next, gamma):
    output = critic.wrapped_forward(input, aj).squeeze()
    target_output = target_critic.wrapped_forward(input_next, aj_next).squeeze()

    td_target = r + gamma * target_output
    critic_loss = loss_criterion(td_target, output)

    return critic_loss


def grad_and_step(net, net_optimizer, obj, type):
    assert type in ['descent', 'ascent']

    net_optimizer.zero_grad()
    net_grad = torch.autograd.grad(obj, list(net.parameters()), retain_graph=True)
    net_params = list(net.parameters())
    for layer in range(len(net_params)):
        if type == 'descent':
            net_params[layer].grad = net_grad[layer]
        else:
            net_params[layer].grad = - net_grad[layer]
        net_params[layer].grad.data.clamp_(-1, 1)
    net_optimizer.step()
    return


class sender_class(object):
    def __init__(self, config, device, id=0):
        self.name = name = 'sender'
        self.config = config
        self.device = device
        self.id = id
        self.dim_action = dim_action = config.env.dim_action
        self.epsilon = config.sender.epsilon_greedy

        self.nj = config.env.nj

        # Gi(s,aj)
        self.critic_Gi = critic_embedding(config.n_channels.obs_sender + 1, 1, config, belongto=name,
                                          name='critic_Gi', device=device)
        # Gj(s,aj), all in one
        self.critic_Gj = critic_embedding(config.n_channels.obs_sender + 1, self.nj, config, belongto=name,
                                          name='critic_Gj', device=device)
        # phi(sigma|s)
        self.signaling_net = signaling_net(config, device=device)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_Gi_optimizer = torch.optim.Adam(self.critic_Gi.parameters(), config.sender.lr_critic_Gi)
        self.critic_Gj_optimizer = torch.optim.Adam(self.critic_Gj.parameters(), config.sender.lr_critic_Gj)
        self.signaling_optimizer = torch.optim.Adam(self.signaling_net.parameters(), config.sender.lr_signal)

        # target critics
        self.target_critic_Gi = critic_embedding(config.n_channels.obs_sender + 1, 1, config, belongto=name,
                                                 name='target_critic_Gi', device=device)
        self.target_critic_Gi.load_state_dict(self.critic_Gi.state_dict())
        self.target_critic_Gj = critic_embedding(config.n_channels.obs_sender + 1, self.nj, config, belongto=name,
                                                 name='target_critic_Gj', device=device)
        self.target_critic_Gj.load_state_dict(self.critic_Gj.state_dict())

        self.temperature = 1

        import itertools
        action_list = [[a for a in range(dim_action)] for _ in range(self.nj)]
        self.a_joint_table = torch.tensor(tuple(itertools.product(*action_list)), dtype=torch.float32, device=device)

    def build_connection(self, receiver_list):
        self.receiver_list = receiver_list

    def calculate_2critics_loss(self, batch):
        ri = batch.data[batch.name_dict['ri']]
        rj = batch.data[batch.name_dict['rj']]
        obs_sender = batch.data[batch.name_dict['obs_sender']]
        obs_sender_next = batch.data[batch.name_dict['obs_sender_next']]
        aj = batch.data[batch.name_dict['a']]
        aj_next = batch.data[batch.name_dict['a_next']]

        critic_loss_Gi = sender_calculate_critic_loss(self.critic_Gi, self.target_critic_Gi, self.critic_loss_criterion,
                                                      ri, input=obs_sender, aj=aj, input_next=obs_sender_next,
                                                      aj_next=aj_next, gamma=self.config.sender.gamma)

        critic_loss_Gj = sender_calculate_critic_loss(self.critic_Gj, self.target_critic_Gj, self.critic_loss_criterion,
                                                      rj, input=obs_sender, aj=aj, input_next=obs_sender_next,
                                                      aj_next=aj_next, gamma=self.config.sender.gamma)
        return critic_loss_Gi, critic_loss_Gj

    def softupdate_2target_critics(self):
        tau = self.config.nn.target_critic_tau
        for tar, cur in zip(self.target_critic_Gi.parameters(), self.critic_Gi.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        for tar, cur in zip(self.target_critic_Gj.parameters(), self.critic_Gj.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        return

    def calculate_for_updating(self, batch):
        critic_loss_Gi, critic_loss_Gj = self.calculate_2critics_loss(batch)
        gradeta = self.calculate_gradeta(batch)

        return critic_loss_Gi, critic_loss_Gj, gradeta

    def update(self, critic_loss_Gi, critic_loss_Gj, gradeta):

        grad_and_step(self.critic_Gi, self.critic_Gi_optimizer, critic_loss_Gi, 'descent')
        grad_and_step(self.critic_Gj, self.critic_Gj_optimizer, critic_loss_Gj, 'descent')

        self.softupdate_2target_critics()

        self.signaling_optimizer.zero_grad()
        params = list(self.signaling_net.parameters())
        for i in range(len(list(self.signaling_net.parameters()))):
            params[i].grad = - gradeta[i]  # gradient ascent
            params[i].grad.data.clamp_(-1, 1)
        self.signaling_optimizer.step()

    def send_message(self, obs):
        batch_size = len(obs)
        logits, phi = self.signaling_net(obs)
        phi = (1 - self.epsilon) * phi + self.epsilon / phi.shape[0]
        sample = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)
        message = sample.view(batch_size, self.config.env.nj, self.config.env.map_height, self.config.env.map_width)
        return message, phi

    def calculate_gradeta(self, batch):
        height, width = self.config.env.map_height, self.config.env.map_width

        obs_sender = batch.data[batch.name_dict['obs_sender']]
        aj = batch.data[batch.name_dict['a']]
        pij = batch.data[batch.name_dict['pi']]

        batch_size, nj, dim_a = pij.size()
        aj_view = torch.flatten(aj)
        pij_view = pij.view(batch_size * nj, dim_a)
        pij_aj = pij_view[range(len(aj_view)), aj_view]
        pij_aj = pij_aj.view(batch_size, nj)

        phi = batch.data[batch.name_dict['phi']]
        # phi_np = np.array(phi.detach())
        sigma = message = batch.data[batch.name_dict['message']]
        phi_view = phi.view(batch_size * nj, -1)
        sigma_view = sigma.view(batch_size * nj, height, width)

        # phi_np = phi.detach().numpy()

        sigma_flatten = sigma.view(sigma_view.shape[0], -1)
        idx_flatten = torch.nonzero(sigma_flatten)[:, 1]
        phi_sigma = phi_view[range(idx_flatten.shape[0]), idx_flatten]
        phi_sigma = phi_sigma.view(batch_size, nj)

        ''' SG (Signaling Gradient) '''
        # s, aj
        Gi = self.critic_Gi.wrapped_forward(obs_sender, aj)

        log_phi_sigma = torch.log(phi_sigma)
        log_pij_aj = torch.log(pij_aj)

        # tuning for gumbel-softmax
        term = torch.mean(Gi.detach() * (log_phi_sigma
                                         + log_pij_aj * self.config.sender.coe_for_recovery_fromgumbel))

        gradeta = torch.autograd.grad(term, list(self.signaling_net.parameters()), retain_graph=True)
        gradeta_flatten = flatten_layers(gradeta)

        ''' BCE Obedience Constraint (Lagrangian) '''
        sigma_counterfactual_x = torch.randint(height, size=(batch_size, nj)).to(self.device)
        sigma_counterfactual_y = torch.randint(width, size=(batch_size, nj)).to(self.device)
        sigma_counterfactual_int = sigma_counterfactual_x * width + sigma_counterfactual_y
        sigma_counterfactual = torch.nn.functional.one_hot(sigma_counterfactual_int, num_classes=height * width) \
            .view(batch_size, nj, height, width).to(torch.float32)

        # oj
        obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']]
        obs_receiver = obs_and_message_receiver[:, :, 0:self.config.n_channels.obs_and_message_receiver - 1, :, :]
        if obs_receiver.size()[2] == 0:
            obs_receiver = None

        if obs_receiver is not None:
            obs_and_message_counterfactual_receiver = torch.cat([obs_receiver, sigma_counterfactual.unsqueeze(dim=2)],
                                                                dim=2)
        else:
            obs_and_message_counterfactual_receiver = sigma_counterfactual.unsqueeze(dim=2)

        pij_counterfactual_list = []
        for j in range(nj):
            _, pij_counterfactual = self.receiver_list[j].choose_action(
                obs_and_message_counterfactual_receiver[:, j, :, :, :])
            pij_counterfactual_list.append(pij_counterfactual.unsqueeze(dim=1))
        pij_counterfactual_all = torch.cat(pij_counterfactual_list, dim=1)
        pij_delta = pij - pij_counterfactual_all

        pij_joint = pij_delta[:, 0, :]
        for j in range(1, nj):
            pij_joint = torch.bmm(pij_joint.unsqueeze(dim=-1), pij_delta[:, j, :].unsqueeze(dim=1)).view(batch_size, -1)
        # pij_np = pij_joint.detach().numpy()

        a_table_size = self.a_joint_table.size()[0]
        a_joint_table_repeat = self.a_joint_table.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        obs_sender_repeat = obs_sender.unsqueeze(dim=1).repeat(1, a_table_size, 1, 1, 1)
        # size_common = a_joint_table_repeat.size()
        # size_temp = []
        a_joint_table_repeat_view = a_joint_table_repeat.view(batch_size * a_table_size, -1)
        obs_sender_repeat_view = obs_sender_repeat.view(batch_size * a_table_size, 2 * nj + 1, height, width)

        # s, aj
        Gj_table = self.critic_Gj.wrapped_forward(obs_sender_repeat_view, a_joint_table_repeat_view)
        Gj_view = Gj_table.view(batch_size, a_table_size, nj)
        # Vj = self.calculate_v(self.critic_Gj, obs_sender, phi, obs_receiver)
        # advantage_j_table = Gj_table - Vj.unsqueeze(dim=1).repeat(1, self.dim_action)
        term = torch.prod(phi_sigma, dim=1).unsqueeze(dim=-1).repeat(1, nj) \
               * torch.sum(pij_joint.unsqueeze(dim=-1).repeat(1, 1, nj).detach() * Gj_view.detach(), dim=1)

        constraint_left = torch.mean(term, dim=0)
        flag = constraint_left < 0
        mask = flag.to(int).to(self.device)
        constraint_left_needgrad = torch.sum(constraint_left * mask)

        gradeta_constraint_term = torch.autograd.grad(constraint_left_needgrad, list(self.signaling_net.parameters()),
                                                      retain_graph=True)
        gradeta_constraint_flatten = flatten_layers(gradeta_constraint_term)

        if self.config.sender.sender_objective_alpha >= 1:
            gradeta_flatten = gradeta_flatten / self.config.sender.sender_objective_alpha + gradeta_constraint_flatten
        elif 0 <= self.config.sender.sender_objective_alpha < 1:
            gradeta_flatten = gradeta_flatten + self.config.sender.sender_objective_alpha * gradeta_constraint_flatten
        else:
            # raise IOError
            pass

        # reform to be in original shape
        gradeta_flatten = gradeta_flatten.squeeze()
        gradeta = []
        idx = 0
        for layerl in self.signaling_net.parameters():
            len_layerl = 1
            for i in layerl.shape:
                len_layerl *= i
            gradeta_layerl_section = gradeta_flatten[idx:idx + len_layerl]
            gradeta_layerl = gradeta_layerl_section.view(layerl.shape)
            gradeta.append(gradeta_layerl)
            idx += len_layerl

        return gradeta

    def save_models(self):
        self.critic_Gi.save_checkpoint()
        self.critic_Gj.save_checkpoint()
        self.target_critic_Gi.save_checkpoint()
        self.target_critic_Gj.save_checkpoint()
        self.signaling_net.save_checkpoint()

    def load_models(self):
        self.critic_Gi.load_checkpoint()
        self.critic_Gj.load_checkpoint()
        self.target_critic_Gi.load_checkpoint()
        self.target_critic_Gj.load_checkpoint()
        self.signaling_net.load_checkpoint()


class receiver_class(object):
    def __init__(self, config, device, id=1):
        self.config = config
        self.device = device
        self.id = id
        self.name = name = 'receiver_' + str(id)

        self.dim_action = dim_action = config.env.dim_action

        self.critic_Qj = critic(config.n_channels.obs_and_message_receiver, dim_action, config, belongto=name,
                                device=device)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_Qj_optimizer = torch.optim.Adam(self.critic_Qj.parameters(), config.receiver.lr_critic_Gj)

        self.actor = actor(config.n_channels.obs_and_message_receiver, config, belongto=name, device=device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.receiver.lr_actor)
        self.gamma = config.receiver.gamma

        self.target_critic_Qj = critic(config.n_channels.obs_and_message_receiver, dim_action, config, belongto=name,
                                       name='target_critic', device=device)
        self.target_critic_Qj.load_state_dict(self.critic_Qj.state_dict())

    def build_connection(self, sender):
        self.sender = sender

    def choose_action(self, input):
        return self.actor.get_a_and_pi(input)

    def calculate_v_foractor(self, critic, input_critic, pi):
        q_table = critic(input_critic)
        assert q_table.shape[1] == self.config.env.dim_action
        v = torch.sum(q_table * pi, dim=1)
        return v

    def calculate_actorobj_and_entropy(self, critic, input_critic, a, pi, ):
        q_table = critic(input_critic)
        q = q_table[range(len(a)), a]
        v = self.calculate_v_foractor(critic, input_critic, pi)
        advantage = q - v
        pi_a = pi[range(len(a)), a]
        actor_obj = advantage.detach() * torch.log(pi_a)
        actor_obj_mean = torch.mean(actor_obj)

        entropy = -torch.sum(pi_a * torch.log(pi_a))

        return actor_obj_mean, entropy

    def calculate_for_updating(self, batch):
        rj = batch.data[batch.name_dict['rj']][:, self.id - 1]
        obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']][:, self.id - 1, :, :, :]
        obs_and_message_receiver_next = batch.data[
                                            batch.name_dict['obs_and_message_receiver_next']][:, self.id - 1, :, :, :]
        aj = batch.data[batch.name_dict['a']][:, self.id - 1]
        aj_next = batch.data[batch.name_dict['a_next']][:, self.id - 1]

        critic_loss_Qj = calculate_critic_loss(self.critic_Qj, self.target_critic_Qj,
                                               self.critic_loss_criterion, r=rj,
                                               input=obs_and_message_receiver, a=aj,
                                               input_next=obs_and_message_receiver_next,
                                               a_next=aj_next,
                                               gamma=self.gamma)

        # critic, input_critic, a, pi
        # pij = batch.data[batch.name_dict['pij']]
        _, pij = self.choose_action(obs_and_message_receiver)
        actor_obj_mean, entropy = self.calculate_actorobj_and_entropy(self.critic_Qj, obs_and_message_receiver, aj, pij)

        return critic_loss_Qj, actor_obj_mean, entropy

    def update(self, critic_loss_Qj, actor_obj_mean, entropy):
        grad_and_step(self.actor, self.actor_optimizer, actor_obj_mean + self.config.receiver.entropy_coe * entropy,
                      'ascent')
        grad_and_step(self.critic_Qj, self.critic_Qj_optimizer, critic_loss_Qj, 'descent')

        return

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_Qj.save_checkpoint()
        self.target_critic_Qj.save_checkpoint()

    def load_models(self, path=None):
        self.actor.load_checkpoint(path)
        self.critic_Qj.load_checkpoint(path)
        self.target_critic_Qj.load_checkpoint(path)
