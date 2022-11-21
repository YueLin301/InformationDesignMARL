import torch
from exp_recommendation.rec_utils import set_net_params, int_to_onehot, flatten_layers


class pro_class():
    def __init__(self, config):
        self.name = 'pro'
        self.config = config

        # q(s,a), rather than q(s,sigma)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.double), torch.nn.Tanh(),
            torch.nn.Linear(in_features=2, out_features=1, bias=False, dtype=torch.double)
        )
        self.signaling_net = torch.nn.Sequential(
            # input: one hot; output: signaling 0/1 prob. distribution
            torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.double),
            torch.nn.Softmax(dim=-1)
        )

        if config.pro.initialize:
            # set_net_params(self.critic, params=config.pro.critic_pro_params)
            set_net_params(self.signaling_net, params=config.pro.signaling_params)

        self.critic_loss_criterion = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.pro.lr_pro_critic)
        self.signaling_optimizer = torch.optim.Adam(self.signaling_net.parameters(), config.pro.lr_signal)

        self.temperature = 0.1
        self.softmax_forGumble = torch.nn.Softmax(dim=-1)
        self.message_table = torch.tensor([0, 1], dtype=torch.double)

    def build_connection(self, hr):
        self.hr = hr

    def update_c(self, buffer):
        critic_loss = 0
        for transition in buffer:
            obs = transition.obs_pro
            obs_int = 0 if obs < 0.5 else 1
            obs_onehot = int_to_onehot(obs_int)
            r = transition.reward_pro  # reward_pro

            a_int_hr = transition.a_int_hr
            a_onehot_hr = int_to_onehot(a_int_hr)

            obs_and_a_onehot = torch.cat([obs_onehot, a_onehot_hr])

            q = self.critic(obs_and_a_onehot).squeeze()
            q_next = 0
            td_target = r + q_next
            td_target = torch.tensor(td_target, dtype=torch.double)  # 没梯度的，没事

            critic_loss_i = self.critic_loss_criterion(td_target, q)  # 没梯度的，没事
            critic_loss = critic_loss + critic_loss_i
        critic_loss = critic_loss / len(buffer)

        self.critic_optimizer.zero_grad()
        critic_grad = torch.autograd.grad(critic_loss, list(self.critic.parameters()), retain_graph=True)
        critic_params = list(self.critic.parameters())
        for layer in range(len(critic_params)):
            critic_params[layer].grad = critic_grad[layer]
            critic_params[layer].grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        return

    def gumbel_sample(self, dim=2):
        u = torch.rand(dim, dtype=torch.double)
        g = - torch.log(-torch.log(u))
        return g

    def send_message(self, obs):
        obs_int = 0 if obs < 0.5 else 1
        if not self.config.pro.fixed_signaling_scheme:
            obs_onehot = int_to_onehot(obs_int)
            phi_current = self.signaling_net(obs_onehot)
        else:
            phi_current = self.config.pro.signaling_scheme[obs_int]

        g = self.gumbel_sample(dim=2)
        logits_forGumbel = (torch.log(phi_current) + g) / self.temperature
        message_onehot = self.softmax_forGumble(logits_forGumbel)  # one hot
        message = torch.einsum('i,i->', self.message_table, message_onehot).unsqueeze(dim=0)

        return message_onehot, phi_current, message

    def update_infor_design(self, buffer):
        constraint_left = 0
        gradeta_flatten = 0
        for transition in buffer:
            obs = transition.obs_pro
            obs_int = 0 if obs < 0.5 else 1
            obs_onehot = int_to_onehot(obs_int)
            a = transition.a_int_hr  # a_int_hr
            r_hr = transition.reward_hr  # reward_hr
            sigma = transition.message_pro  # message_pro
            phi_sigma = transition.message_prob_pro[int(sigma)]  # message_prob_pro[int(sigma)]
            pi_at = transition.a_prob_hr[a]  # a_prob_hr[at]

            gradeta_phi_sigma = torch.autograd.grad(phi_sigma, list(self.signaling_net.parameters()), retain_graph=True)
            gradeta_phi_sigma_flatten = flatten_layers(gradeta_phi_sigma, dim=1)  # (n, 1)

            gradeta_pi_at = torch.autograd.grad(pi_at, list(self.signaling_net.parameters()), retain_graph=True)
            gradeta_pi_at_flatten = flatten_layers(gradeta_pi_at, dim=1)

            a_int_hr = transition.a_int_hr
            a_onehot_hr = int_to_onehot(a_int_hr)
            obs_and_a_onehot = torch.cat([obs_onehot, a_onehot_hr])
            q = self.critic(obs_and_a_onehot).squeeze()

            gradeta_flatten_i = q * (
                    pi_at * gradeta_phi_sigma_flatten * self.temperature
                    + phi_sigma * gradeta_pi_at_flatten
            )
            gradeta_flatten = gradeta_flatten + gradeta_flatten_i

            # Constraints, Lagrangian
            # 采样更新state，其他照常求梯度
            a_couterfactual = 1 - a
            r_couterfactual = self.config.env.rewardmap_HR[obs_int][a_couterfactual]

            constraint_left_i = phi_sigma * pi_at * (r_hr - r_couterfactual)
            constraint_left = constraint_left + constraint_left_i
        constraint_left = constraint_left / len(buffer)
        gradeta_flatten = gradeta_flatten / len(buffer)

        if constraint_left > self.config.pro.constraint_right:
            gradeta_constraint = torch.autograd.grad(constraint_left, list(self.signaling_net.parameters()),
                                                     retain_graph=True)
            gradeta_constraint_flatten = flatten_layers(gradeta_constraint, dim=1)
            gradeta_flatten = gradeta_flatten + self.config.pro.sender_objective_alpha * gradeta_constraint_flatten

        # 返回为原来梯度的样子
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
            params[i].grad = - gradeta[i]  # 梯度上升
            params[i].grad.data.clamp_(-1, 1)
        self.signaling_optimizer.step()

        return


class hr_class():
    def __init__(self, config):
        self.name = 'hr'
        self.config = config

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.double), torch.nn.Tanh(),
            torch.nn.Linear(in_features=2, out_features=1, bias=False, dtype=torch.double)
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.double),
            torch.nn.Softmax(dim=-1)
        )

        if config.hr.initialize:
            # set_net_params(self.critic, params=config.hr.critic_params)
            set_net_params(self.actor, params=config.hr.actor_params)

        self.critic_loss_criterion = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.hr.lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.hr.lr_actor)

        self.epsilon = config.hr.epsilon_start

    def build_connection(self, pro):
        self.pro = pro

    def choose_action(self, message, using_epsilon=False):
        pi = self.actor(message)
        if using_epsilon:
            pi = (1 - self.epsilon) * pi + self.epsilon / 2
        distribution = torch.distributions.Categorical(pi)
        a = distribution.sample([1])
        return a, pi, distribution.log_prob(a),

    def calculate_v(self, message, pi):
        v = 0
        for a_idx in range(len(pi)):
            pi_a_i = pi[a_idx].detach().unsqueeze(dim=0)
            a_onehot = int_to_onehot(a_idx)
            message_and_a_i = torch.cat([message, a_onehot])
            q_i = self.critic(message_and_a_i)
            v = v + pi_a_i * q_i
        return v

    def update_ac(self, buffer):
        critic_loss = 0
        actor_obj = 0
        for transition in buffer:
            a_onehot_hr = int_to_onehot(transition.a_int_hr)
            message_and_a = torch.cat([transition.message_onehot_pro, a_onehot_hr])

            q = self.critic(message_and_a).squeeze()
            q_next = 0

            td_target = transition.reward_hr + q_next
            td_target = torch.tensor(td_target, dtype=torch.double)  # 没梯度的，没事

            critic_loss_i = self.critic_loss_criterion(td_target, q)  # 没梯度的，没事
            critic_loss = critic_loss + critic_loss_i

            if self.config.train.GAE_term == 'TD-error':
                td_error = td_target - q
                actor_obj_i = td_error * transition.a_logprob_hr
            elif self.config.train.GAE_term == 'advantage':
                v = self.calculate_v(transition.message_onehot_pro, transition.a_prob_hr)
                advantage = q - v
                actor_obj_i = advantage * transition.a_logprob_hr
            else:
                raise NotImplementedError

            actor_obj = actor_obj + actor_obj_i

        critic_loss = critic_loss / len(buffer)
        actor_obj = actor_obj / len(buffer)

        '''更新'''
        if not self.config.hr.fixed_policy:
            self.actor_optimizer.zero_grad()
            actor_grad = torch.autograd.grad(actor_obj, list(self.actor.parameters()), retain_graph=True)
            actor_params = list(self.actor.parameters())
            for layer in range(len(actor_params)):
                actor_params[layer].grad = - actor_grad[layer]
                actor_params[layer].grad.data.clamp_(-1, 1)
            self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_grad = torch.autograd.grad(critic_loss, list(self.critic.parameters()), retain_graph=True)
        critic_params = list(self.critic.parameters())
        for layer in range(len(critic_params)):
            critic_params[layer].grad = critic_grad[layer]
            critic_params[layer].grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        return
