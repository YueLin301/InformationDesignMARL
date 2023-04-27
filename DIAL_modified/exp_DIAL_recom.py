import sys

sys.path.append('../')

import torch
from torch.nn.utils import clip_grad_norm_
from DIAL_origin.agent import DRU

from exp_recommendation.rec_utils import init_wandb
from tqdm import tqdm


class pro_DIAL():
    def __init__(self, config):
        self.name = 'pro'
        self.config = config

        self.embedding_layer = torch.nn.Embedding(num_embeddings=2,  # how many values in a dim (input)
                                                  embedding_dim=2,  # output_dim
                                                  dtype=torch.double)

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=4, bias=False, dtype=torch.double), torch.nn.Tanh(),
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.double)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.pro.lr_critic)
        self.dru = DRU(sigma=2)

    def build_connection(self, hr):
        self.hr = hr

    def send_message(self, obs, train_mode=True):
        obs_emb = self.embedding_layer(obs)
        q = self.critic(obs_emb)
        message = self.dru.forward(q, train_mode)
        return message


class hr_DIAL():
    def __init__(self, config):
        self.name = 'hr'
        self.config = config

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=4, bias=False, dtype=torch.double), torch.nn.Tanh(),
            torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.double)
        )

        self.critic_loss_criterion = torch.nn.MSELoss(reduction='mean')
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config.hr.lr_critic)

    def build_connection(self, pro):
        self.pro = pro

    def choose_action(self, message):
        q = self.critic(message)

        pi_raw = torch.zeros_like(q, dtype=torch.double)
        _, max_idx = torch.max(q, dim=1)
        pi_raw[torch.arange(q.shape[0]), max_idx] = 1

        # epsilon-greedy
        n_action = q.shape[1]
        p_eps = self.config.hr.eps / n_action
        pi = pi_raw * (1 - 2 * p_eps) + p_eps

        a = torch.multinomial(pi, num_samples=1, replacement=True, )[:, 0]
        return a, q

    def update_for_all(self, r, q_a):
        self.critic_optimizer.zero_grad()
        self.pro.critic_optimizer.zero_grad()

        td_loss = self.critic_loss_criterion(r, q_a)
        td_loss.backward()

        clip_grad_norm_(list(self.critic.parameters()), max_norm=1)
        clip_grad_norm_(list(self.pro.critic.parameters()), max_norm=1)

        self.critic_optimizer.step()
        self.pro.critic_optimizer.step()


class buffer_class(object):
    def __init__(self, ):
        self.reset()

    def reset(self):
        self.obs_pro, self.message_pro, \
        self.a_int_hr, self.reward_pro, self.reward_hr = [None] * 5

        self.values = None

    def set_values(self, buffer_values):
        self.obs_pro, self.message_pro, self.a_int_hr, \
        self.reward_pro, self.reward_hr = buffer_values

        self.values = buffer_values

    def get_values(self):
        return self.obs_pro, self.message_pro, self.a_int_hr, \
               self.reward_pro, self.reward_hr


def run_an_episode(env, pro, hr, ):
    obs_pro = env.reset()
    message_pro = pro.send_message(obs_pro, train_mode=True)
    a_int_hr, q = hr.choose_action(message_pro)
    reward_pro, reward_hr = env.step(obs_pro, a_int_hr)

    buffer = buffer_class()
    buffer.set_values([obs_pro, message_pro, a_int_hr,
                       reward_pro, reward_hr])

    return buffer, q


def train(config, env, pro, hr, group_name, seed=None):
    print('----------------------------------------')

    chart_name_list, run_handle = init_wandb(group_name, config.pro.sender_objective_alpha)
    if not seed is None:
        run_handle.tags = run_handle.tags + (str(seed),)
    print('Start training.')

    pbar = tqdm(total=config.train.n_episodes)

    i_episode = 0
    while i_episode < config.train.n_episodes:
        buffer, q = run_an_episode(env, pro, hr)
        plot_with_wandb(chart_name_list, buffer, i_episode, config.env.sample_n_students)
        i_episode += config.env.sample_n_students
        pbar.update(config.env.sample_n_students)

        hr.update_for_all(buffer.reward_hr, q[torch.arange(q.shape[0]), buffer.a_int_hr])

    run_handle.finish()
    pbar.close()
    return


def plot_with_wandb(chart_name_list, batch, i_episode, env_sample_n_students):
    entry = dict(zip(chart_name_list, [0] * len(chart_name_list)))

    reward_pro_tensor = batch.reward_pro.detach()
    reward_hr_tensor = batch.reward_hr.detach()
    reward_tot_tensor = reward_pro_tensor + reward_hr_tensor

    entry['reward_sender'] = float(torch.mean(reward_pro_tensor))
    entry['reward_receiver'] = float(torch.mean(reward_hr_tensor))
    entry['social_welfare'] = float(torch.mean(reward_tot_tensor))

    wandb.log(entry, step=i_episode * env_sample_n_students)


if __name__ == '__main__':
    myseeds = [i for i in range(10)]

    from DIAL_modified.config_rec import config
    from env import recommendation
    import wandb
    from exp_recommendation.mykey import wandb_login_key
    from exp_recommendation.rec_utils import set_seed

    wandb.login(key=wandb_login_key)
    device = torch.device('cpu')
    for seed in myseeds:
        set_seed(seed)

        pro = pro_DIAL(config)
        hr = hr_DIAL(config)
        pro.build_connection(hr)
        hr.build_connection(pro)
        env = recommendation.recommendation_env(config.env, device)

        train(config, env, pro, hr, group_name="DIAL_test1", seed=seed)
