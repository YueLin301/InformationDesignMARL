import torch
from exp_recommendation.rec_utils import set_net_params, int_to_onehot, flatten_layers


class pro_formal_constrained():
    def __init__(self, config, device):
        self.name = 'pro'
        self.config = config
        self.device = device

        # G^i(s,a), rather than Q(s,sigma)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.float64), torch.nn.Tanh(),
            torch.nn.Linear(in_features=2, out_features=1, bias=False, dtype=torch.float64)
        ).to(device)
        # G^j(s,a), rather than Q(sigma,a)
        self.critic_forhr = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.float64), torch.nn.Tanh(),
            torch.nn.Linear(in_features=2, out_features=1, bias=False, dtype=torch.float64)
        ).to(device)
        self.signaling_net = torch.nn.Sequential(
            # input: one hot; output: signaling 0/1 prob. distribution
            torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.float64),
            torch.nn.Softmax(dim=-1)
        ).to(device)

        if config.pro.initialize:
            set_net_params(self.signaling_net, params=config.pro.signaling_params)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.pro.lr_pro_critic)
        self.critic_optimizer_forhr = torch.optim.Adam(self.critic_forhr.parameters(), config.pro.lr_pro_critic)
        self.signaling_optimizer = torch.optim.Adam(self.signaling_net.parameters(), config.pro.lr_signal)

        self.temperature = 1
        self.softmax_forGumble = torch.nn.Softmax(dim=-1)
        self.message_table = torch.tensor([0, 1], dtype=torch.float64, device=self.device)

        self.sender_objective_alpha = self.config.pro.sender_objective_alpha

    def build_connection(self, hr):
        self.hr = hr

    def update_c(self, buffer):
        obs = buffer.obs_pro
        obs_onehot = int_to_onehot(obs, k=2, device=self.device)
        a_int_hr = buffer.a_int_hr
        a_onehot_hr = int_to_onehot(a_int_hr, k=2, device=self.device)
        obs_and_a_onehot = torch.cat([obs_onehot, a_onehot_hr], dim=1)

        r = buffer.reward_pro  # reward_pro
        G = self.critic(obs_and_a_onehot).squeeze()
        G_next = 0
        td_target = r + G_next
        critic_loss = self.critic_loss_criterion(td_target, G)

        self.critic_optimizer.zero_grad()
        critic_grad = torch.autograd.grad(critic_loss, list(self.critic.parameters()), retain_graph=True)
        critic_params = list(self.critic.parameters())
        for layer in range(len(critic_params)):
            critic_params[layer].grad = critic_grad[layer]
            critic_params[layer].grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        r_hr = buffer.reward_hr
        Gj = self.critic_forhr(obs_and_a_onehot).squeeze()
        Gj_next = 0
        td_target_j = r_hr + Gj_next
        critic_loss_forhr = self.critic_loss_criterion(td_target_j, Gj)

        self.critic_optimizer_forhr.zero_grad()
        critic_grad_forhr = torch.autograd.grad(critic_loss_forhr, list(self.critic_forhr.parameters()),
                                                retain_graph=True)
        critic_params = list(self.critic_forhr.parameters())
        for layer in range(len(critic_params)):
            critic_params[layer].grad = critic_grad_forhr[layer]
            critic_params[layer].grad.data.clamp_(-1, 1)
        self.critic_optimizer_forhr.step()

        return

    def gumbel_sample(self, dim=2):
        u = torch.rand(dim, dtype=torch.float64)
        g = - torch.log(-torch.log(u))
        return g

    def my_gumbel_softmax(self, phi):
        g = self.gumbel_sample(dim=2)
        logits_forGumbel = (torch.log(phi) + g) / self.temperature
        message_onehot = self.softmax_forGumble(logits_forGumbel)  # onehot
        return message_onehot

    def send_message(self, obs_list):
        obs_list = [obs_list] if isinstance(obs_list, int) else obs_list
        if not self.config.pro.fixed_signaling_scheme:
            obs_onehot = int_to_onehot(obs_list, k=2, device=self.device)
            phi_current = self.signaling_net(obs_onehot)
        else:
            phi_current = self.config.pro.signaling_scheme[obs_list]

        logits = torch.log(phi_current)
        message_onehot = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)
        message = torch.einsum('i,ji->j', self.message_table, message_onehot)

        # # for tuning config.pro.coe_for_recovery_fromgumbel
        # temp11 = torch.autograd.grad(phi_current[0, 0], list(self.signaling_net.parameters()), retain_graph=True)
        # temp12 = torch.autograd.grad(phi_current[0, 1], list(self.signaling_net.parameters()), retain_graph=True)
        # temp2 = torch.autograd.grad(message[0], phi_current, retain_graph=True)
        # temp3 = torch.autograd.grad(message_onehot[0, 0], list(self.signaling_net.parameters()), retain_graph=True)
        # temp4 = torch.autograd.grad(message_onehot[0, 1], list(self.signaling_net.parameters()), retain_graph=True)
        # temp5 = torch.autograd.grad(message[0], list(self.signaling_net.parameters()), retain_graph=True)

        return message_onehot, phi_current, message

    def update_infor_design(self, buffer):
        obs = buffer.obs_pro
        obs_onehot = int_to_onehot(obs, k=2, device=self.device)
        a = buffer.a_int_hr
        sigma = buffer.message_pro
        sigma_int = torch.round(sigma).long()
        phi_sigma = buffer.message_prob_pro[range(len(sigma)), sigma_int]
        pi_at = buffer.a_prob_hr[range(len(sigma)), a]

        ''' SG (Signaling Gradient) '''
        a_int_hr = buffer.a_int_hr
        a_onehot_hr = int_to_onehot(a_int_hr, k=2, device=self.device)
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

        a1 = torch.tensor([1, 0], dtype=torch.float64, device=self.device).unsqueeze(dim=0).repeat(
            self.config.env.sample_n_students, 1)
        a2 = torch.tensor([0, 1], dtype=torch.float64, device=self.device).unsqueeze(dim=0).repeat(
            self.config.env.sample_n_students, 1)
        obs_and_a1 = torch.cat([obs_onehot, a1], dim=1)
        obs_and_a2 = torch.cat([obs_onehot, a2], dim=1)

        Gj_obs_and_a1 = self.critic_forhr(obs_and_a1)
        Gj_obs_and_a2 = self.critic_forhr(obs_and_a2)
        Gj_obs_and_a = torch.cat([Gj_obs_and_a1, Gj_obs_and_a2], dim=1)

        constraint_left = torch.mean(phi_sigma * torch.sum((pi - pi_counterfactual) * Gj_obs_and_a, dim=1))
        if constraint_left < self.config.pro.constraint_right:
            constraint_term = torch.mean(
                phi_sigma * torch.sum(
                    (pi.detach() - pi_counterfactual.detach())
                    * Gj_obs_and_a.detach(), dim=1))
            gradeta_constraint_term = torch.autograd.grad(constraint_term,
                                                          list(self.signaling_net.parameters()), retain_graph=True)
            gradeta_constraint_flatten = flatten_layers(gradeta_constraint_term, 0)
            gradeta_flatten = gradeta_flatten + self.sender_objective_alpha * gradeta_constraint_flatten

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

        # if i_episode > 1 / 3 * self.config.train.n_episodes and self.sender_objective_alpha > 0:
        #     self.sender_objective_alpha -= self.config.pro.sender_objective_alpha / \
        #                                    (1 / 3 * self.config.train.n_episodes / self.config.env.sample_n_students)

        return
