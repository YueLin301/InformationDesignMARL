import torch
from exp_harvest.harvest_utils import flatten_layers, generate_receiver_obs_and_message_counterfactual
from exp_harvest.network import actor, critic, signaling_net


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


class agent_base_class():
    def __init__(self, type, config, device, id):
        self.config = config
        self.device = device
        self.id = id
        assert type in ['sender', 'receiver']
        self.type = type

        actor_n_channels = config.n_channels.obs_sender + config.n_channels.message \
            if type == 'sender' else config.n_channels.obs_receiver + config.n_channels.message
        lr_actor = config.sender.lr_actor if type == 'sender' else config.receiver.lr_actor

        self.actor = actor(actor_n_channels, config, belongto=type, device=device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr_actor)

        self.gamma = config.sender.gamma if type == 'sender' else config.receiver.gamma

    def build_connection(self, partner):
        if self.type == 'receiver':
            self.sender = partner
        else:
            self.receiver = partner

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


class sender_class(agent_base_class):
    def __init__(self, config, device, id=0):
        self.name = name = 'sender'
        super().__init__(name, config, device, id)

        self.dim_action = dim_action = config.env.dim_action

        # s, ai, aj
        self.critic_Gi = critic(config.n_channels.obs_sender, dim_action ** 2, config, belongto=name, name='critic_Gi',
                                device=device)
        # s, aj
        self.critic_Gj = critic(config.n_channels.obs_sender, dim_action, config, belongto=name, name='critic_Gj',
                                device=device)
        # s, sigma, ai
        self.critic_foractor = critic(config.n_channels.obs_sender + config.n_channels.message, dim_action, config,
                                      belongto=name, name='critic_foractor', device=device)
        self.signaling_net = signaling_net(config, device=device)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_Gi_optimizer = torch.optim.Adam(self.critic_Gi.parameters(), config.sender.lr_critic_Gi)
        self.critic_Gj_optimizer = torch.optim.Adam(self.critic_Gj.parameters(), config.sender.lr_critic_Gj)
        self.critic_foractor_optimizer = torch.optim.Adam(self.critic_foractor.parameters(),
                                                          config.sender.lr_critic_foractor)
        self.signaling_optimizer = torch.optim.Adam(self.signaling_net.parameters(), config.sender.lr_signal)

        # target critics
        self.target_critic_Gi = critic(config.n_channels.obs_sender, dim_action ** 2, config, belongto=name,
                                       name='target_critic_Gi', device=device)
        self.target_critic_Gi.load_state_dict(self.critic_Gi.state_dict())
        self.target_critic_Gj = critic(config.n_channels.obs_sender, dim_action, config, belongto=name,
                                       name='target_critic_Gj', device=device)
        self.target_critic_Gj.load_state_dict(self.critic_Gj.state_dict())
        self.target_critic_foractor = critic(config.n_channels.obs_sender + config.n_channels.message, dim_action,
                                             config, belongto=name, name='target_critic_foractor', device=device)
        self.target_critic_foractor.load_state_dict(self.critic_foractor.state_dict())

        self.temperature = 1
        self.message_table = torch.tensor([0, 1], dtype=torch.double).to(self.device)  # from onehot to 1

    def calculate_3critics_loss(self, batch):
        ri = batch.data[batch.name_dict['ri']]
        rj = batch.data[batch.name_dict['rj']]
        obs_sender = batch.data[batch.name_dict['obs_sender']]
        obs_sender_next = batch.data[batch.name_dict['obs_sender_next']]

        message = batch.data[batch.name_dict['message']]
        obs_and_message_sender = torch.cat([obs_sender, message], dim=1)

        message_next = batch.data[batch.name_dict['message_next']]
        obs_and_message_sender_next = torch.cat([obs_sender_next, message_next], dim=1)

        ai = batch.data[batch.name_dict['ai']]
        ai_next = batch.data[batch.name_dict['ai_next']]
        aj = batch.data[batch.name_dict['aj']]
        aj_next = batch.data[batch.name_dict['aj_next']]
        a_idx = ai * self.dim_action + aj
        a_idx_next = ai_next * self.dim_action + aj_next

        critic_loss_Gi = calculate_critic_loss(self.critic_Gi, self.target_critic_Gi, self.critic_loss_criterion,
                                               ri, input=obs_sender, a=a_idx, input_next=obs_sender_next,
                                               a_next=a_idx_next, gamma=self.gamma)
        critic_loss_Gj = calculate_critic_loss(self.critic_Gj, self.target_critic_Gj, self.critic_loss_criterion,
                                               rj, input=obs_sender, a=aj, input_next=obs_sender_next, a_next=aj_next,
                                               gamma=self.gamma)
        critic_loss_foractor = calculate_critic_loss(self.critic_foractor, self.target_critic_foractor,
                                                     self.critic_loss_criterion, ri, input=obs_and_message_sender, a=ai,
                                                     input_next=obs_and_message_sender_next, a_next=ai_next,
                                                     gamma=self.gamma)
        return critic_loss_Gi, critic_loss_Gj, critic_loss_foractor

    def softupdate_3target_critics(self):
        tau = self.config.nn.target_critic_tau
        for tar, cur in zip(self.target_critic_Gi.parameters(), self.critic_Gi.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        for tar, cur in zip(self.target_critic_Gj.parameters(), self.critic_Gj.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        for tar, cur in zip(self.target_critic_foractor.parameters(), self.critic_foractor.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        return

    def calculate_for_updating(self, batch):
        critic_loss_Gi, critic_loss_Gj, critic_loss_foractor = self.calculate_3critics_loss(batch)

        # critic, input_critic, a, pi
        # input_critic <= s, sigma, ai
        obs_sender = batch.data[batch.name_dict['obs_sender']]
        message = batch.data[batch.name_dict['message']]
        obs_and_message_sender = torch.cat([obs_sender, message], dim=1)
        ai = batch.data[batch.name_dict['ai']]
        # pii = batch.data[batch.name_dict['pii']]
        _, pii = self.choose_action(obs_and_message_sender)
        actor_obj_mean, entropy = self.calculate_actorobj_and_entropy(self.critic_foractor,
                                                                      input_critic=obs_and_message_sender, a=ai, pi=pii)

        gradeta = self.calculate_gradeta(batch)

        return critic_loss_Gi, critic_loss_Gj, critic_loss_foractor, actor_obj_mean, entropy, gradeta

    def update(self, critic_loss_Gi, critic_loss_Gj, critic_loss_foractor, actor_obj_mean, entropy, gradeta):

        grad_and_step(self.critic_Gi, self.critic_Gi_optimizer, critic_loss_Gi, 'descent')
        grad_and_step(self.critic_Gj, self.critic_Gj_optimizer, critic_loss_Gj, 'descent')
        grad_and_step(self.critic_foractor, self.critic_foractor_optimizer, critic_loss_foractor, 'descent')

        grad_and_step(self.actor, self.actor_optimizer, actor_obj_mean + self.config.sender.entropy_coe * entropy,
                      'ascent')

        self.softupdate_3target_critics()

        self.signaling_optimizer.zero_grad()
        params = list(self.signaling_net.parameters())
        for i in range(len(list(self.signaling_net.parameters()))):
            params[i].grad = - gradeta[i]  # gradient ascent
            params[i].grad.data.clamp_(-1, 1)
        self.signaling_optimizer.step()

    def send_message(self, obs):
        phi = self.signaling_net(obs)
        logits = torch.log(phi)
        sample = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)

        batch_size = len(obs)
        message_flatten = torch.einsum('i,kji->kj', self.message_table, sample)
        message = message_flatten.view(batch_size, 1, self.config.env.obs_height, self.config.env.obs_width)

        # for tuning config.pro.coe_for_recovery_fromgumbel
        # temp11 = torch.autograd.grad(phi_current[0, 0], list(self.signaling_net.parameters()), retain_graph=True)
        # temp12 = torch.autograd.grad(phi_current[0, 1], list(self.signaling_net.parameters()), retain_graph=True)
        # temp2 = torch.autograd.grad(message[0], phi_current, retain_graph=True)
        # temp5 = torch.autograd.grad(message[0], list(self.signaling_net.parameters()), retain_graph=True)

        return message, phi

    def calculate_gradeta(self, batch):
        obs_sender = batch.data[batch.name_dict['obs_sender']]
        # phi = batch.data[batch.name_dict['phi']]
        _, phi = self.send_message(obs_sender)
        sigma = message = batch.data[batch.name_dict['message']]
        obs_and_message_sender = torch.cat([obs_sender, message], dim=1)

        phi_flatten = phi.view(-1, phi.shape[-1])
        sigma_flatten = torch.flatten(sigma).long()
        phi_sigma_flatten = phi_flatten[range(len(sigma_flatten)), sigma_flatten]
        phi_sigma = phi_sigma_flatten.view(phi.shape[:-1])

        ai = batch.data[batch.name_dict['ai']]
        aj = batch.data[batch.name_dict['aj']]

        # pii = batch.data[batch.name_dict['pii']]
        # pij = batch.data[batch.name_dict['pij']]
        _, pii = self.choose_action(obs_and_message_sender)
        obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']]
        _, pij = self.receiver.choose_action(obs_and_message_receiver)

        pii_ai = pii[range(len(ai)), ai]
        pij_aj = pij[range(len(aj)), aj]

        ''' SG (Signaling Gradient) '''
        # s, ai, aj
        Gi_table = self.critic_Gi(obs_sender)
        a_idx = self.config.env.dim_action * ai + aj
        Gi = Gi_table[range(len(a_idx)), a_idx]

        log_phi_sigma = torch.log(phi_sigma)
        log_phi_sigma_sum = torch.sum(log_phi_sigma, dim=1)

        log_pii_ai = torch.log(pii_ai)
        log_pij_aj = torch.log(pij_aj)

        # tuning for gumbel-softmax
        term = torch.mean(Gi.detach() * (log_phi_sigma_sum
                                         + log_pii_ai
                                         + log_pij_aj))

        gradeta = torch.autograd.grad(term, list(self.signaling_net.parameters()), retain_graph=True)
        gradeta_flatten = flatten_layers(gradeta)

        ''' BCE Obedience Constraint (Lagrangian) '''
        sigma_counterfactual = torch.round(torch.rand_like(sigma))  # negative sampling
        obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']]
        obs_and_message_counterfactual_receiver = generate_receiver_obs_and_message_counterfactual(
            obs_and_message_receiver, sigma_counterfactual)

        _, pij_counterfactual = self.receiver.choose_action(obs_and_message_counterfactual_receiver)

        # s, aj
        Gj_table = self.critic_Gj(obs_sender)
        # term = phi_sigma * torch.sum((pij - pij_counterfactual) * Gj_table)
        term1 = phi_sigma * torch.sum(pij.detach() * Gj_table.detach(), dim=1).unsqueeze(dim=1)
        term2 = phi_sigma.detach() * torch.sum(pij * Gj_table.detach(), dim=1).unsqueeze(dim=1)

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
        gradeta_flatten = gradeta_flatten + self.config.sender.sender_objective_alpha * gradeta_constraint_flatten

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
        self.actor.save_checkpoint()
        self.critic_Gi.save_checkpoint()
        self.critic_Gj.save_checkpoint()
        self.critic_foractor.save_checkpoint()
        self.target_critic_Gi.save_checkpoint()
        self.target_critic_Gj.save_checkpoint()
        self.target_critic_foractor.save_checkpoint()
        self.signaling_net.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_Gi.load_checkpoint()
        self.critic_Gj.load_checkpoint()
        self.target_critic_Gi.load_checkpoint()
        self.target_critic_Gj.load_checkpoint()
        self.signaling_net.load_checkpoint()


class receiver_class(agent_base_class):
    def __init__(self, config, device, id=1):
        self.name = name = 'receiver'
        super().__init__(name, config, device, id)

        self.dim_action = dim_action = config.env.dim_action

        # obs_receiver, sigma, aj
        self.critic_Qj = critic(config.n_channels.obs_receiver + config.n_channels.message, dim_action, config,
                                belongto=name, device=device)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_Qj_optimizer = torch.optim.Adam(self.critic_Qj.parameters(), config.receiver.lr_critic_Gj)

        self.target_critic_Qj = critic(config.n_channels.obs_receiver + config.n_channels.message, dim_action, config,
                                       belongto=name, name='target_critic', device=device)
        self.target_critic_Qj.load_state_dict(self.critic_Qj.state_dict())

    def calculate_for_updating(self, batch):
        rj = batch.data[batch.name_dict['rj']]
        obs_and_message_receiver = batch.data[batch.name_dict['obs_and_message_receiver']]
        obs_and_message_receiver_next = batch.data[batch.name_dict['obs_and_message_receiver_next']]
        aj = batch.data[batch.name_dict['aj']]
        aj_next = batch.data[batch.name_dict['aj_next']]

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
