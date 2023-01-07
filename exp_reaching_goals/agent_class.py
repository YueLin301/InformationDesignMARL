import torch
from exp_reaching_goals.reaching_goals_utils import flatten_layers
from exp_reaching_goals.network import actor, critic, signaling_net

import numpy as np


def calculate_critic_loss(critic, target_critic, loss_criterion, r, input, a, input_next, a_next, gamma):
    output_table = critic(input)
    target_output_table = target_critic(input_next)
    output = output_table[range(len(a)), a]
    target_output = target_output_table[range(len(a)), a_next]

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

        # Gi(s,aj)
        self.critic_Gi = critic(config.n_channels.obs_sender, dim_action, config, belongto=name, name='critic_Gi',
                                device=device)
        # Gi(s,aj)
        self.critic_Gj = critic(config.n_channels.obs_sender, dim_action, config, belongto=name, name='critic_Gj',
                                device=device)
        # phi(sigma|s)
        self.signaling_net = signaling_net(config, device=device)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_Gi_optimizer = torch.optim.Adam(self.critic_Gi.parameters(), config.sender.lr_critic_Gi)
        self.critic_Gj_optimizer = torch.optim.Adam(self.critic_Gj.parameters(), config.sender.lr_critic_Gj)
        self.signaling_optimizer = torch.optim.Adam(self.signaling_net.parameters(), config.sender.lr_signal)

        # target critics
        self.target_critic_Gi = critic(config.n_channels.obs_sender, dim_action, config, belongto=name,
                                       name='target_critic_Gi', device=device)
        self.target_critic_Gi.load_state_dict(self.critic_Gi.state_dict())
        self.target_critic_Gj = critic(config.n_channels.obs_sender, dim_action, config, belongto=name,
                                       name='target_critic_Gj', device=device)
        self.target_critic_Gj.load_state_dict(self.critic_Gj.state_dict())

        self.temperature = 1

    def build_connection(self, receiver):
        self.receiver = receiver

    def calculate_v(self, critic, input_critic, phi, obs_receiver):
        # v(s) = \sum_sigma phi(sigma|s) * \sum_a pi(a|sigma) * Gi(s,a)
        batch_size = phi.shape[0]
        message_dim = phi.shape[1]
        all_message = torch.nn.functional.one_hot(torch.arange(message_dim)) \
            .view(message_dim,
                  self.config.env.map_height,
                  self.config.env.map_width) \
            .unsqueeze(dim=0).repeat(batch_size, 1, 1, 1).unsqueeze(dim=2).to(self.device)

        obs_receiver = obs_receiver.repeat(1, message_dim, 1, 1).unsqueeze(dim=2)
        obs_and_message_receiver = torch.cat([obs_receiver, all_message], dim=2)

        obs_and_message_receiver_flatten = obs_and_message_receiver.view(batch_size * message_dim, 2,
                                                                         obs_and_message_receiver.shape[-2],
                                                                         obs_and_message_receiver.shape[-1])

        _, pi_flatten = self.receiver.choose_action(obs_and_message_receiver_flatten)
        pi = pi_flatten.view(obs_and_message_receiver.shape[0], obs_and_message_receiver.shape[1], pi_flatten.shape[-1])
        pi_sum_all_message = torch.sum(pi * phi.unsqueeze(dim=2).repeat(1, 1, pi.shape[-1]), dim=1)

        g_table = critic(input_critic)
        v = torch.sum(g_table * pi_sum_all_message, dim=1)
        return v

    def calculate_2critics_loss(self, batch):
        ri = batch.data[batch.name_dict['ri']]
        rj = batch.data[batch.name_dict['rj']]
        obs_sender = batch.data[batch.name_dict['obs_sender']]
        obs_sender_next = batch.data[batch.name_dict['obs_sender_next']]
        aj = batch.data[batch.name_dict['a']]
        aj_next = batch.data[batch.name_dict['a_next']]

        critic_loss_Gi = calculate_critic_loss(self.critic_Gi, self.target_critic_Gi, self.critic_loss_criterion,
                                               ri, input=obs_sender, a=aj, input_next=obs_sender_next, a_next=aj_next,
                                               gamma=self.config.sender.gamma)
        critic_loss_Gj = calculate_critic_loss(self.critic_Gj, self.target_critic_Gj, self.critic_loss_criterion,
                                               rj, input=obs_sender, a=aj, input_next=obs_sender_next, a_next=aj_next,
                                               gamma=self.config.sender.gamma)
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
        message = sample.view(batch_size, 1, self.config.env.map_height, self.config.env.map_width)
        return message, phi

    def calculate_gradeta(self, batch):
        obs_sender = batch.data[batch.name_dict['obs_sender']]
        aj = batch.data[batch.name_dict['a']]
        pij = batch.data[batch.name_dict['pi']]
        pij_aj = pij[range(len(aj)), aj]

        phi = batch.data[batch.name_dict['phi']]
        # phi_np = np.array(phi.detach())
        sigma = message = batch.data[batch.name_dict['message']]
        sigma_flatten = sigma.view(sigma.shape[0], -1)
        idx_flatten = torch.nonzero(sigma_flatten)[:, 1]
        phi_sigma = phi[range(idx_flatten.shape[0]), idx_flatten]

        ''' SG (Signaling Gradient) '''
        # s, aj
        Gi_table = self.critic_Gi(obs_sender)
        Gi = Gi_table[range(len(aj)), aj]
        obs_receiver = obs_sender[:, 0:1:, :,
                       :]  # This is for advantage. The other way is to make the receiver to tell the results, in which the sender doesn't need to have access to this var.
        Vi = self.calculate_v(self.critic_Gi, obs_sender, phi, obs_receiver)
        advantage_i = Gi - Vi

        log_phi_sigma = torch.log(phi_sigma)
        log_pij_aj = torch.log(pij_aj)

        # tuning for gumbel-softmax
        term = torch.mean(advantage_i.detach() * (log_phi_sigma
                                                  + log_pij_aj * self.config.sender.coe_for_recovery_fromgumbel))
        # term1 = torch.mean(Gi.detach() * log_phi_sigma)
        # term2 = torch.mean(Gi.detach() * log_pij_aj)
        # gradeta1 = torch.autograd.grad(term1, list(self.signaling_net.parameters()), retain_graph=True)
        # gradeta2 = torch.autograd.grad(term2, list(self.signaling_net.parameters()), retain_graph=True)

        gradeta = torch.autograd.grad(term, list(self.signaling_net.parameters()), retain_graph=True)
        gradeta_flatten = flatten_layers(gradeta)

        ''' BCE Obedience Constraint (Lagrangian) '''
        if not self.config.env.aligned_object:
            batch_len = aj.shape[0]
            sigma_counterfactual_index_flatten = torch.randint(self.config.env.map_height * self.config.env.map_width,
                                                               size=(batch_len,)).to(self.device)  # negative sampling
            sigma_counterfactual_index = [
                torch.floor(sigma_counterfactual_index_flatten / self.config.env.map_height).long().unsqueeze(dim=0),
                (sigma_counterfactual_index_flatten % self.config.env.map_width).unsqueeze(dim=0)]
            sigma_counterfactual_index = torch.cat(sigma_counterfactual_index).to(self.device)
            sigma_counterfactual = torch.zeros(batch_len, self.config.env.map_height, self.config.env.map_width,
                                               dtype=torch.double).to(self.device)
            sigma_counterfactual[range(batch_len), sigma_counterfactual_index[0], sigma_counterfactual_index[1]] = 1
            # sigma_counterfactual_np = np.array(sigma_counterfactual)
            sigma_counterfactual = sigma_counterfactual.unsqueeze(dim=1)
            obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']]
            obs_and_message_counterfactual_receiver = torch.cat([obs_and_message_receiver[:, 0:1, :, :],
                                                                 sigma_counterfactual], dim=1)
            _, pij_counterfactual = self.receiver.choose_action(obs_and_message_counterfactual_receiver)

            # s, aj
            Gj_table = self.critic_Gj(obs_sender)
            # Vj = self.calculate_v(self.critic_Gj, obs_sender, phi, obs_receiver)
            # advantage_j_table = Gj_table - Vj.unsqueeze(dim=1).repeat(1, self.dim_action)
            term1 = phi_sigma * torch.sum(pij.detach() * Gj_table.detach(), dim=1).unsqueeze(dim=1)
            term2 = phi_sigma.detach() * torch.sum(pij * Gj_table.detach(), dim=1).unsqueeze(dim=1)
            # term1 = phi_sigma * torch.sum(pij.detach() * advantage_j_table.detach(), dim=1).unsqueeze(dim=1)
            # term2 = phi_sigma.detach() * torch.sum(pij * advantage_j_table.detach(), dim=1).unsqueeze(dim=1)

            # every pixel
            term1_sum = torch.sum(term1, dim=1)
            term2_sum = torch.sum(term2, dim=1)

            term1_mean = torch.mean(term1_sum)
            term2_mean = torch.mean(term2_sum)

            gradeta_constraint_term1 = torch.autograd.grad(term1_mean, list(self.signaling_net.parameters()),
                                                           retain_graph=True)
            gradeta_constraint_term2 = torch.autograd.grad(term2_mean, list(self.signaling_net.parameters()),
                                                           retain_graph=True)

            gradeta_constraint_term_1st_flatten = flatten_layers(gradeta_constraint_term1)
            gradeta_constraint_term_2nd_flatten = flatten_layers(
                gradeta_constraint_term2) * self.config.sender.coe_for_recovery_fromgumbel

            gradeta_constraint_flatten = gradeta_constraint_term_1st_flatten \
                                         + gradeta_constraint_term_2nd_flatten
            if self.config.sender.sender_objective_alpha >= 1:
                gradeta_flatten = gradeta_flatten / self.config.sender.sender_objective_alpha + gradeta_constraint_flatten
            elif 0 <= self.config.sender.sender_objective_alpha < 1:
                gradeta_flatten = gradeta_flatten + self.config.sender.sender_objective_alpha * gradeta_constraint_flatten
            else:
                raise IOError

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
        self.name = name = 'receiver'
        self.config = config
        self.device = device
        self.id = id

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

        # entropy = -torch.sum(pi * torch.log(pi))
        entropy = -torch.sum(pi_a * torch.log(pi_a))

        return actor_obj_mean, entropy

    def calculate_for_updating(self, batch):
        rj = batch.data[batch.name_dict['rj']]
        obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']]
        obs_and_message_receiver_next = batch.data[batch.name_dict['obs_and_message_receiver_next']]
        aj = batch.data[batch.name_dict['a']]
        aj_next = batch.data[batch.name_dict['a_next']]

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

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_Qj.load_checkpoint()
        self.target_critic_Qj.load_checkpoint()
