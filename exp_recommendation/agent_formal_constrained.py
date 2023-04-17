import torch
from exp_recommendation.rec_utils import set_net_params, int_to_onehot, flatten_layers


class pro_formal_constrained():
    def __init__(self, config):
        self.name = 'pro'
        self.config = config

        # G^i(s,a), rather than Q(s,sigma)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.double), torch.nn.Tanh(),
            torch.nn.Linear(in_features=2, out_features=1, bias=False, dtype=torch.double)
        )
        # G^j(s,a), rather than Q(sigma,a)
        self.critic_forhr = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.double), torch.nn.Tanh(),
            torch.nn.Linear(in_features=2, out_features=1, bias=False, dtype=torch.double)
        )
        # phi(sigma|s), distribution
        self.signaling_net = torch.nn.Sequential(
            # input: one hot; output: signaling 0/1 prob. distribution
            torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.double),
            torch.nn.Softmax(dim=-1)
        )

        # # lambda(sigma_table); sigma' is sigma_t from tau
        # self.lambda_net = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=2, out_features=4, bias=False, dtype=torch.double), torch.nn.Tanh(),
        #     torch.nn.Linear(in_features=4, out_features=1, bias=False, dtype=torch.double), torch.nn.ReLU()
        # )
        self.lambda_table = torch.rand(2)

        if config.pro.initialize:
            set_net_params(self.signaling_net, params=config.pro.signaling_params)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.pro.lr_pro_critic)
        self.critic_optimizer_forhr = torch.optim.Adam(self.critic_forhr.parameters(), config.pro.lr_pro_critic)
        self.signaling_optimizer = torch.optim.Adam(self.signaling_net.parameters(), config.pro.lr_signal)

        self.temperature = 1
        self.softmax_forGumble = torch.nn.Softmax(dim=-1)

        self.message_table = torch.tensor([0, 1], dtype=torch.double)
        message_table_int = torch.tensor([0, 1], dtype=torch.long)
        self.message_table_onehot = int_to_onehot(message_table_int, k=2)
        self.aj_table_int = torch.tensor([0, 1], dtype=torch.long)
        # self.aj_table_onehot = int_to_onehot(aj_table_int, k=2)

        self.sender_objective_alpha = self.config.pro.sender_objective_alpha

        self.debug_counter = 0

    def build_connection(self, hr):
        self.hr = hr

    def update_c(self, buffer):
        obs = buffer.obs_pro
        obs_onehot = int_to_onehot(obs, k=2)
        a_int_hr = buffer.a_int_hr
        a_onehot_hr = int_to_onehot(a_int_hr, k=2)
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
        u = torch.rand(dim, dtype=torch.double)
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
            obs_onehot = int_to_onehot(obs_list, k=2)
            phi_current = self.signaling_net(obs_onehot)
        else:
            phi_current = self.config.pro.signaling_scheme[obs_list]

        logits = torch.log(phi_current)
        message_onehot = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)
        message = torch.einsum('i,ji->j', self.message_table, message_onehot)

        return message_onehot, phi_current, message

    def estimate_OC_left(self, sigma, sigma_cf, phi_t_list):

        return

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

        ''' BCE - Obedience Constraint (Dual Gradient Descent) '''
        # T: buffer length, the amount of time-steps (s_t)
        # M: the amount of signals
        # N: the amount of actions
        phi_sigma_st = self.signaling_net(obs_onehot)  # (T,M)
        pi_a_sigma_table = self.hr.actor(self.message_table_onehot)  # (M,N)

        # sigma_cf_table = torch.rand_like(self.message_table_onehot)
        sigma_cf_table_int = torch.randint(2, self.message_table.shape)
        sigma_cf_table = int_to_onehot(sigma_cf_table_int, 2)
        pi_a_sigmacf_table = self.hr.actor(sigma_cf_table)  # (M,N)

        pi_a_sigma_delta = pi_a_sigma_table - pi_a_sigmacf_table
        Pr_st_sigma_a = torch.einsum('ij, jk -> ijk', phi_sigma_st, pi_a_sigma_delta.detach())  # (T,M,N)

        # # sigma' (sigma_cf, sigma_t form tau)
        # pi_t = buffer.a_prob_hr  # (T,N)
        # Pr_st_sigmacf_a = torch.einsum('ij,ik->ijk', phi_sigma_st, pi_t.detach())  # (T,M,N)
        # Pr_st_sigma_a_delta = Pr_st_sigma_a - Pr_st_sigmacf_a

        # reshape
        TMN = Pr_st_sigma_a.shape
        Pr_sigma_st_a_delta = torch.transpose(Pr_st_sigma_a, 0, 1) \
            .reshape(TMN[1], TMN[0] * TMN[2])  # (T,M,N) -> (M,T,N) -> (M,T*N)

        # (s,a) onehot -> critic -> w^j(s,a)
        sa_raw = torch.cartesian_prod(obs, self.aj_table_int)
        sa_s_onehot = int_to_onehot(sa_raw[:, 0], k=2)
        sa_a_onehot = int_to_onehot(sa_raw[:, 1], k=2)
        sa_onehot = torch.cat([sa_s_onehot, sa_a_onehot], dim=1)

        wj_st_a = self.critic_forhr(sa_onehot).squeeze()  # (T*N)

        C_sigma_table = torch.einsum('ij,j->i', Pr_sigma_st_a_delta, wj_st_a.detach()) / TMN[0]
        # lambda_table = self.lambda_net(self.message_table_onehot).squeeze()
        # lambda_C = torch.sum(lambda_table.detach() * C_sigma_table)
        lambda_C = torch.sum(self.lambda_table * C_sigma_table)

        gradeta_lambda_C = torch.autograd.grad(lambda_C, list(self.signaling_net.parameters()), retain_graph=True)
        gradeta_lambda_C_flatten = flatten_layers(gradeta_lambda_C, 0)
        gradeta_flatten = gradeta_flatten + self.sender_objective_alpha * gradeta_lambda_C_flatten

        ## lambda update
        lambda_table_temp = self.lambda_table - self.sender_objective_alpha \
                            * (C_sigma_table - self.config.pro.constraint_right)
        flag = lambda_table_temp > 0
        flag = flag.type(torch.double)
        self.lambda_table = lambda_table_temp * flag

        # self.debug_counter += 1
        # if self.debug_counter > 2000:
        #     print('debug haha')

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
