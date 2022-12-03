import torch
from exp_harvest.harvest_utils import flatten_layers

from exp_harvest.network import actor, critic, signaling_net


def update_critic(critic, target_critic, critic_optimizer, loss_criterion, r, input, input_next):
    output = critic(*input)
    target_output = target_critic(input_next)
    td_target = r + target_output
    critic_loss = loss_criterion(td_target, output)

    critic_optimizer.zero_grad()
    critic_grad = torch.autograd.grad(critic_loss, list(critic.parameters()), retain_graph=True)
    critic_params = list(critic.parameters())
    for layer in range(len(critic_params)):
        critic_params[layer].grad = critic_grad[layer]
        critic_params[layer].grad.data.clamp_(-1, 1)
    critic_optimizer.step()
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
        assert len(q_table) == self.config.env.dim_action
        v = torch.sum(q_table * pi)
        return v

    def update_actor(self, critic, input_critic, a, pi, ):
        q_table = critic(input_critic)
        q = q_table[a]
        v = self.calculate_v_foractor(critic, input_critic, pi)
        advantage = q - v
        actor_obj = advantage * torch.log(pi[a])
        actor_obj_mean = torch.mean(actor_obj)

        entropy = -torch.sum(pi[a] * torch.log(pi[a]))

        self.actor_optimizer.zero_grad()
        actor_grad = torch.autograd.grad(actor_obj_mean + self.config.hr.entropy_coe * entropy,
                                         list(self.actor.parameters()), retain_graph=True)
        actor_params = list(self.actor.parameters())
        for layer in range(len(actor_params)):
            actor_params[layer].grad = - actor_grad[layer]
            actor_params[layer].grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        return


class sender_class(agent_base_class):
    def __init__(self, config, device, id=0):
        self.name = name = 'sender'
        super().__init__(name, config, device, id)

        dim_action = config.env.dim_action

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

    def update_2critics(self, ri, input_Gi, input_next_Gi, rj, input_Gj, input_next_Gj):
        update_critic(self.critic_Gi, self.target_critic_Gi, self.critic_Gi_optimizer, self.critic_loss_criterion,
                      ri, input_Gi, input_next_Gi)
        update_critic(self.critic_Gj, self.target_critic_Gj, self.critic_Gj_optimizer, self.critic_loss_criterion,
                      rj, input_Gj, input_next_Gj)
        return

    def softupdate_target_2critics(self):
        tau = self.config.nn.target_critic_tau
        for tar, cur in zip(self.target_critic_Gi.parameters(), self.critic_Gi.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        for tar, cur in zip(self.target_critic_Gj.parameters(), self.critic_Gj.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        return

    def update_ac(self, buffer):
        self.update_2critics(buffer)
        self.update_actor()
        self.softupdate_target_2critics()
        return

    def send_message(self, obs):
        phi = self.signaling_net(obs)
        logits = torch.log(phi)
        sample = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)

        batch_size = len(obs)
        message_flatten = torch.einsum('i,ji->j', self.message_table, sample)
        message = message_flatten.view(batch_size, 1, self.config.env.obs_height, self.config.env.obs_width)

        # for tuning config.pro.coe_for_recovery_fromgumbel
        # temp11 = torch.autograd.grad(phi_current[0, 0], list(self.signaling_net.parameters()), retain_graph=True)
        # temp12 = torch.autograd.grad(phi_current[0, 1], list(self.signaling_net.parameters()), retain_graph=True)
        # temp2 = torch.autograd.grad(message[0], phi_current, retain_graph=True)
        # temp5 = torch.autograd.grad(message[0], list(self.signaling_net.parameters()), retain_graph=True)

        return message, phi

    def update_infor_design(self, buffer):
        obs = buffer.obs_pro
        obs_onehot = int_to_onehot(obs, k=2)
        a = buffer.a_int_hr
        sigma = buffer.message_pro
        sigma_int = torch.round(sigma).long()
        phi_sigma = buffer.message_prob_pro[range(len(sigma)), sigma_int]
        pi_at = buffer.a_prob_hr[range(len(sigma)), a]

        ''' SG (Signaling Gradient) '''
        a_int_hr = buffer.a_int_hr
        a_onehot_hr = int_to_onehot(a_int_hr, k=2)
        obs_and_a_onehot = torch.cat([obs_onehot, a_onehot_hr], dim=1)
        G = self.critic(obs_and_a_onehot).squeeze()

        log_phi_sigma = torch.log(phi_sigma)
        log_pi_at = torch.log(pi_at)

        term_1st = torch.mean(G.detach() * log_phi_sigma)
        term_2nd = torch.mean(G.detach() * log_pi_at)

        G_times_gradeta_log_phi = torch.autograd.grad(term_1st, list(self.signaling_net.parameters()),
                                                      retain_graph=True)
        G_times_gradeta_log_pi = torch.autograd.grad(term_2nd, list(self.signaling_net.parameters()),
                                                     retain_graph=True)

        G_times_gradeta_log_phi_flatten = flatten_layers(G_times_gradeta_log_phi)
        G_times_gradeta_log_pi_flatten = flatten_layers(G_times_gradeta_log_pi)

        gradeta_flatten = G_times_gradeta_log_phi_flatten \
                          + G_times_gradeta_log_pi_flatten * self.config.pro.coe_for_recovery_fromgumbel  # This is for recovering scale from gumbel-softmax process

        ''' BCE Obedience Constraint (Lagrangian) '''
        pi = buffer.a_prob_hr
        sigma_onehot = buffer.message_onehot_pro
        sigma_counterfactual_onehot = 1 - sigma_onehot.detach()
        _, pi_counterfactual, _ = self.hr.choose_action(sigma_counterfactual_onehot)

        a1 = torch.tensor([1, 0], dtype=torch.double).unsqueeze(dim=0).repeat(100, 1)
        a2 = torch.tensor([0, 1], dtype=torch.double).unsqueeze(dim=0).repeat(100, 1)
        obs_and_a1 = torch.cat([obs_onehot, a1], dim=1)
        obs_and_a2 = torch.cat([obs_onehot, a2], dim=1)

        Gj_obs_and_a1 = self.critic_forhr(obs_and_a1)
        Gj_obs_and_a2 = self.critic_forhr(obs_and_a2)
        Gj_obs_and_a = torch.cat([Gj_obs_and_a1, Gj_obs_and_a2], dim=1)

        # constraint_left = torch.mean(phi_sigma * torch.sum((pi - pi_counterfactual) * Gj_obs_and_a, dim=1))
        # if constraint_left < self.config.pro.constraint_right:
        constraint_term_1st = torch.mean(phi_sigma * torch.sum(pi.detach() * Gj_obs_and_a.detach(), dim=1))
        constraint_term_2nd = torch.mean(phi_sigma.detach() * torch.sum(pi * Gj_obs_and_a.detach(), dim=1))

        gradeta_constraint_term_1st = torch.autograd.grad(constraint_term_1st,
                                                          list(self.signaling_net.parameters()), retain_graph=True)
        gradeta_constraint_term_2nd = torch.autograd.grad(constraint_term_2nd,
                                                          list(self.signaling_net.parameters()), retain_graph=True)

        gradeta_constraint_term_1st_flatten = flatten_layers(gradeta_constraint_term_1st, 0)
        gradeta_constraint_term_2nd_flatten = flatten_layers(gradeta_constraint_term_2nd,
                                                             0) * self.config.pro.coe_for_recovery_fromgumbel

        gradeta_constraint_flatten = gradeta_constraint_term_1st_flatten \
                                     + gradeta_constraint_term_2nd_flatten
        gradeta_flatten = gradeta_flatten + self.config.pro.sender_objective_alpha * gradeta_constraint_flatten

        # reform to be in original shape
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

        self.signaling_optimizer.zero_grad()
        params = list(self.signaling_net.parameters())
        for i in range(len(list(self.signaling_net.parameters()))):
            params[i].grad = - gradeta[i]  # gradient ascent
            params[i].grad.data.clamp_(-1, 1)
        self.signaling_optimizer.step()

        return

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_Gi.save_checkpoint()
        self.critic_Gj.save_checkpoint()
        self.target_critic_Gi.save_checkpoint()
        self.target_critic_Gj.save_checkpoint()
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

        dim_action = config.env.dim_action

        # obs_receiver, sigma, aj
        self.critic_Qj = critic(config.n_channels.obs_receiver + config.n_channels.message, dim_action, config,
                                belongto=name, device=device)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_Qj_optimizer = torch.optim.Adam(self.critic_Qj.parameters(), config.receiver.lr_critic_Gj)

        self.target_critic_Qj = critic(config.n_channels.obs_receiver + config.n_channels.message, dim_action, config,
                                       belongto=name, name='target_critic', device=device)
        self.target_critic_Qj.load_state_dict(self.critic_Qj.state_dict())

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_Qj.save_checkpoint()
        self.target_critic_Qj.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_Qj.load_checkpoint()
        self.target_critic_Qj.load_checkpoint()
