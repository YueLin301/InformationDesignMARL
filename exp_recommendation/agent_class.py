import torch
from exp_recommendation.rec_utils import set_net_params, int_to_onehot, flatten_layers

class hr_class():
    def __init__(self, config, device):
        self.name = 'hr'
        self.config = config
        self.device = device

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.float64), torch.nn.Tanh(),
            torch.nn.Linear(in_features=2, out_features=1, bias=False, dtype=torch.float64)
        ).to(device)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.float64),
            torch.nn.Softmax(dim=-1)
        ).to(device)

        if config.hr.initialize:
            # set_net_params(self.critic, params=config.hr.critic_params)
            set_net_params(self.actor, params=config.hr.actor_params)

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.hr.lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.hr.lr_actor)

    def build_connection(self, pro):
        self.pro = pro

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
            a_onehot = int_to_onehot([a_idx] * n_samples, k=2, device=self.device)
            message_and_a_i = torch.cat([message, a_onehot], dim=1)
            q_i = self.critic(message_and_a_i).squeeze(dim=1)
            v = v + pi_a_i * q_i
        return v

    def update_ac(self, buffer):
        a_onehot_hr = int_to_onehot(buffer.a_int_hr, k=2, device=self.device)
        message_and_a = torch.cat([buffer.message_onehot_pro, a_onehot_hr], dim=1)

        q = self.critic(message_and_a).squeeze()
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

        self.critic_optimizer.zero_grad()
        critic_grad = torch.autograd.grad(critic_loss, list(self.critic.parameters()), retain_graph=True)
        critic_params = list(self.critic.parameters())
        for layer in range(len(critic_params)):
            critic_params[layer].grad = critic_grad[layer]
            critic_params[layer].grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        return
