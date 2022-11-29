import torch
from exp_harvest.harvest_utils import set_net_params, int_to_onehot, flatten_layers

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


class sender_class():
    def __init__(self, config, device):
        self.name = 'sender'
        self.config = config
        self.device = device

        self.n_channels = n_channels = config.env.n_channels

        self.actor = actor(config, device=device)
        self.critic_Gi = critic(config, 'G', device=device)  # Gi(s,aj), rather than Q(s,sigma)
        self.critic_Gj = critic(config, 'G', device=device)  # Gj(s,aj), rather than Q(sigma,a)
        self.signaling_net = signaling_net(config, device=device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.sender.lr_actor)
        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_Gi_optimizer = torch.optim.Adam(self.critic_Gi.parameters(), config.sender.lr_critic_Gi)
        self.critic_Gj_optimizer = torch.optim.Adam(self.critic_Gj.parameters(), config.sender.lr_critic_Gj)
        self.signaling_optimizer = torch.optim.Adam(self.signaling_net.parameters(), config.sender.lr_signal)

        self.target_critic_Gi = critic(config, 'G', device=device).to(device)
        self.target_critic_Gi.load_state_dict(self.critic_Gi.state_dict())
        self.target_critic_Gj = critic(config, 'G', device=device).to(device)
        self.target_critic_Gj.load_state_dict(self.critic_Gj.state_dict())

        self.temperature = 1

    def build_connection(self, receiver):
        self.receiver = receiver

    def update_2critics(self):
        update_critic(self.critic_Gi, self.target_critic_Gi, self.critic_Gi_optimizer, self.critic_loss_criterion,
                      r_Gi, input_Gi, input_next_Gi)
        update_critic(self.critic_Gj, self.target_critic_Gj, self.critic_Gj_optimizer, self.critic_loss_criterion,
                      r_Gj, input_Gj, input_next_Gj)
        return

    def softupdate_target_2critics(self):
        tau = self.config.nn.target_critic_tau
        for tar, cur in zip(self.target_critic_Gi.parameters(), self.critic_Gi.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        for tar, cur in zip(self.target_critic_Gj.parameters(), self.critic_Gj.parameters()):
            tar.data.copy_(cur.data * (1.0 - tau) + tar.data * tau)
        return

    def send_message(self, obs_list):
        obs_list = [obs_list] if isinstance(obs_list, int) else obs_list
        obs_onehot = int_to_onehot(obs_list, k=2)
        phi_current = self.signaling_net(obs_onehot)

        logits = torch.log(phi_current)
        message_onehot = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)

        # # for tuning config.pro.coe_for_recovery_fromgumbel
        # temp11 = torch.autograd.grad(phi_current[0, 0], list(self.signaling_net.parameters()), retain_graph=True)
        # temp12 = torch.autograd.grad(phi_current[0, 1], list(self.signaling_net.parameters()), retain_graph=True)
        # temp2 = torch.autograd.grad(message[0], phi_current, retain_graph=True)
        # temp3 = torch.autograd.grad(message_onehot[0, 0], list(self.signaling_net.parameters()), retain_graph=True)
        # temp4 = torch.autograd.grad(message_onehot[0, 1], list(self.signaling_net.parameters()), retain_graph=True)
        # temp5 = torch.autograd.grad(message[0], list(self.signaling_net.parameters()), retain_graph=True)

        return message_onehot, phi_current

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


class receiver_class():
    def __init__(self, config, device):
        self.name = 'receiver'
        self.config = config
        self.device = device

        self.n_channels = config.env.n_channels

        self.actor = actor(config=config, device=device)
        self.critic_Qj = critic(config=config, device=device)  # Q(sigma,aj)
        self.signaling_net = signaling_net(config=config, device=device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.sender.lr_actor)
        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_Qj_optimizer = torch.optim.Adam(self.critic_Qj.parameters(), config.sender.lr_critic_Gj)

    def build_connection(self, sender):
        self.sender = sender

    def choose_action(self, message):
        pi = self.actor(message)
        distribution = torch.distributions.Categorical(pi)
        a = distribution.sample([1]).squeeze(dim=0)
        return a, pi, distribution.log_prob(a).squeeze(dim=0),

    def calculate_v(self, message, pi):
        v = 0
        n_samples = message.shape[0] if len(message.shape) > 1 else 1
        for a_idx in range(2):
            pi_a_i = pi[range(len(pi)), a_idx].detach()
            a_onehot = int_to_onehot([a_idx] * n_samples, k=2)
            message_and_a_i = torch.cat([message, a_onehot], dim=1)
            q_i = self.critic_Qj(message_and_a_i).squeeze(dim=1)
            v = v + pi_a_i * q_i
        return v

    def update_ac(self, buffer):
        a_onehot_hr = int_to_onehot(buffer.a_int_hr, k=2)
        message_and_a = torch.cat([buffer.message_onehot_pro, a_onehot_hr], dim=1)

        q = self.critic_Qj(message_and_a).squeeze()
        q_next = 0
        td_target = buffer.reward_hr + q_next
        critic_loss = self.critic_loss_criterion(td_target, q)

        v = self.calculate_v(buffer.message_onehot_pro, buffer.a_prob_hr)
        if self.config.train.GAE_term == 'TD-error':
            td_error = td_target - v
            actor_obj = td_error * buffer.a_logprob_hr
        elif self.config.train.GAE_term == 'advantage':

            advantage = q - v
            actor_obj = advantage * buffer.a_logprob_hr
        else:
            raise NotImplementedError
        actor_obj_mean = torch.mean(actor_obj)

        entropy = -torch.sum(buffer.a_prob_hr * torch.log(buffer.a_prob_hr))

        if not self.config.hr.fixed_policy:
            self.actor_optimizer.zero_grad()
            actor_grad = torch.autograd.grad(actor_obj_mean + self.config.hr.entropy_coe * entropy,
                                             list(self.actor.parameters()), retain_graph=True)
            actor_params = list(self.actor.parameters())
            for layer in range(len(actor_params)):
                actor_params[layer].grad = - actor_grad[layer]
                actor_params[layer].grad.data.clamp_(-1, 1)
            self.actor_optimizer.step()

        self.critic_Qj_optimizer.zero_grad()
        critic_grad = torch.autograd.grad(critic_loss, list(self.critic_Qj.parameters()), retain_graph=True)
        critic_params = list(self.critic_Qj.parameters())
        for layer in range(len(critic_params)):
            critic_params[layer].grad = critic_grad[layer]
            critic_params[layer].grad.data.clamp_(-1, 1)
        self.critic_Qj_optimizer.step()

        return
