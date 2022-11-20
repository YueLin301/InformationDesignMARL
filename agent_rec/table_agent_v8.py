import torch
import numpy as np

from env import recommendation
from env.recommendation import student_charc_maptoword, professor_action_maptoword, HR_action_maptoword
from utils.configdict import ConfigDict

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline  # 曲线插值
from scipy.signal import savgol_filter

config_env = ConfigDict()
config_env.prob_good = 1 / 3
config_env.num_sample = 1
config_env.reward_magnification_factor = 1
config_env.have_sender = True

# config_env.rewardmap_professor = [[0, 1],
#                                   [0, 1]]
# config_env.rewardmap_HR = [[0, -1],
#                            [0, 1]]

config_env.rewardmap_professor = [[-1, 11],
                                  [-1, 11]]
config_env.rewardmap_HR = [[-1, -12],
                           [-1, 10]]

# config_env.rewardmap_professor = [[-10, 10],
#                                   [-10, 10]]
# config_env.rewardmap_HR = [[-10, -30],
#                            [-10, 20]]

# config_env.fixed_signaling_scheme = False
config_env.fixed_signaling_scheme = True
if config_env.fixed_signaling_scheme:
    # 完全如实发消息
    # config_env.signaling_scheme = torch.tensor([[1, 0],
    #                                             [0, 1]], dtype=torch.double)
    # 完全乱发消息
    # config_env.signaling_scheme = torch.tensor([[0.5, 0.5],
    #                                             [0.5, 0.5]], dtype=torch.double)
    # 理论最优
    config_env.signaling_scheme = torch.tensor([[0.5, 0.5],
                                                [0, 1]], dtype=torch.double)
    lr_signal = 0
else:
    lr_signal = 1e-3
    # lr_signal = 6.1e-3
    # lr_signal = 1e-2

config_env.initialize = False
# config_env.initialize = True
if config_env.initialize:
    # 直接是均衡点
    config_env.critic_pro_params = torch.tensor([[1, 1, 1, 1]], dtype=torch.double).unsqueeze(dim=0)
    config_env.signaling_params = torch.tensor([[5, 0],
                                                [5, 5]], dtype=torch.double).unsqueeze(dim=0)
    config_env.critic_params = torch.tensor([[-5, 5, 10, 5]], dtype=torch.double).unsqueeze(dim=0)  # 这个不是准确的
    config_env.actor_params = torch.tensor([[5, 0],
                                            [0, 5]], dtype=torch.double).unsqueeze(dim=0)

    # 均衡点旁边
    # config_env.critic_pro_params = torch.tensor([[1, 1, 1, 1]], dtype=torch.double).unsqueeze(dim=0)
    # config_env.signaling_params = torch.tensor([[5, 2],
    #                                             [5, 5]], dtype=torch.double).unsqueeze(dim=0)
    # config_env.critic_params = torch.tensor([[-11, 5, 10, 5]], dtype=torch.double).unsqueeze(dim=0)  # 这个不是准确的
    # config_env.actor_params = torch.tensor([[5, 5],
    #                                         [5, 5]], dtype=torch.double).unsqueeze(dim=0)

lr_pro_critic = 1e-3

lr_critic = 5e-3
# lr_critic = 1e-2

# lr_actor = 3e-3
lr_actor = 5e-3
# lr_actor = 1e-2


constraint_right = 0  # 大于等于，改成等于，所以这个epsilon就不必了
sender_objective_alpha = 0.9039306640625  # 这是拉格朗日的那个lambda

# train_n_episodes = 5e3
# train_n_episodes = 1e4
# train_n_episodes = 3e4
train_n_episodes = 1e5
# train_n_episodes = 5e5
# train_n_episodes = 1e6

myseed = 0
np.random.seed(myseed)  # numpy的随机种子
torch.manual_seed(myseed)  # torch的随机种子
torch.cuda.manual_seed(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_net_params(net, params):
    net_params_list = list(net.parameters())
    for i in range(len(net_params_list)):
        net_params_list[i].data = params[i]
    return


def int_to_onehot(variable_int):
    variable_onehot = [0, 0]
    variable_onehot[variable_int] = 1
    variable_onehot = torch.tensor(variable_onehot, dtype=torch.double)
    return variable_onehot


class pro_class():
    def __init__(self):
        # q(s,a), 不是q(s,sigma)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=1, bias=False, dtype=torch.double),
        )
        self.signaling_net = torch.nn.Sequential(
            # input: one hot; 输出: 发信号0/1的概率分布，用softmax
            torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.double),
            torch.nn.Softmax(dim=-1)
        )
        if config_env.initialize:
            set_net_params(self.critic, params=config_env.critic_pro_params)
            set_net_params(self.signaling_net, params=config_env.signaling_params)

        self.critic_loss_criterion = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr_pro_critic)
        self.signaling_optimizer = torch.optim.Adam(self.signaling_net.parameters(), lr_signal)

        self.temperature = 0.1
        self.softmax_forGumble = torch.nn.Softmax(dim=-1)
        self.message_table = torch.tensor([0, 1], dtype=torch.double)

    def build_connection(self, hr):
        self.hr = hr

    def update_c(self, transition):
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

        critic_loss = self.critic_loss_criterion(td_target, q)  # 没梯度的，没事

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
        if not config_env.fixed_signaling_scheme:
            obs_onehot = int_to_onehot(obs_int)
            phi_current = self.signaling_net(obs_onehot)
        else:
            phi_current = config_env.signaling_scheme[obs_int]

        g = self.gumbel_sample(dim=2)
        logits_forGumbel = (torch.log(phi_current) + g) / self.temperature
        message_onehot = self.softmax_forGumble(logits_forGumbel)  # one hot
        message = torch.einsum('i,i->', self.message_table, message_onehot).unsqueeze(dim=0)

        return message_onehot, phi_current, message

    def flatten_layers(self, gradtensor, dim=0):
        gradtensor_flatten = torch.flatten(gradtensor[0])
        for layerl in range(1, len(gradtensor)):
            temp = torch.flatten(gradtensor[layerl])
            gradtensor_flatten = torch.cat([gradtensor_flatten, temp])
        gradtensor_flatten = gradtensor_flatten.unsqueeze(dim=dim)

        return gradtensor_flatten

    def update_infor_design(self, transition):
        obs = transition.obs_pro
        obs_int = 0 if obs < 0.5 else 1
        obs_onehot = int_to_onehot(obs_int)
        at = transition.a_int_hr  # a_int_hr
        # r_pro = transition.reward_pro  # reward_pro
        r_hr = transition.reward_hr  # reward_hr
        sigma = transition.message_pro  # message_pro
        phi_sigma = transition.message_prob_pro[int(sigma)]  # message_prob_pro[int(sigma)]
        pi_at = transition.a_prob_hr[at]  # a_prob_hr[at]

        gradeta_phi_sigma = torch.autograd.grad(phi_sigma, list(self.signaling_net.parameters()), retain_graph=True)
        gradeta_phi_sigma_flatten = self.flatten_layers(gradeta_phi_sigma, dim=1)  # (n, 1)

        gradeta_pi_at = torch.autograd.grad(pi_at, list(self.signaling_net.parameters()), retain_graph=True)
        gradeta_pi_at_flatten = self.flatten_layers(gradeta_pi_at, dim=1)

        # old_hr_gradeta_v_flatten = self.flatten_layers(self.hr.gradeta_v, dim=1)  # (n, 1)
        # old_hr_gradtheta_logpi_flatten = self.flatten_layers(self.hr.gradtheta_logpi, dim=0)  # (1, m)
        # gradtheta_pi = torch.autograd.grad(pi_at, list(self.hr.actor.parameters()), retain_graph=True)
        # gradtheta_pi_flatten = self.flatten_layers(gradtheta_pi, dim=1)  # (m,1)

        a_int_hr = transition.a_int_hr
        a_onehot_hr = int_to_onehot(a_int_hr)
        obs_and_a_onehot = torch.cat([obs_onehot, a_onehot_hr])
        q = self.critic(obs_and_a_onehot).squeeze()

        gradeta_flatten = q * (
                pi_at * gradeta_phi_sigma_flatten * self.temperature
                + phi_sigma * gradeta_pi_at_flatten
            # - phi_sigma * old_hr_gradeta_v_flatten @ old_hr_gradtheta_logpi_flatten @ gradtheta_pi_flatten
        )

        # Constraints, Lagrangian
        # 采样更新state，其他照常求梯度
        a_onehot = int_to_onehot(at)
        q_hr = self.hr.critic(torch.cat([transition.message_onehot_pro, a_onehot])).squeeze()
        a_couterfactual = 1 - at
        a_couterfactual_onehot = int_to_onehot(a_couterfactual)
        q_hr_counterfactual = self.hr.critic(
            torch.cat([transition.message_onehot_pro, a_couterfactual_onehot])).squeeze()


        constraint_left = phi_sigma * pi_at * (q_hr - q_hr_counterfactual)

        if constraint_left > constraint_right:
            gradeta_constraint_flatten = (q_hr - q_hr_counterfactual) * (
                    pi_at * gradeta_phi_sigma_flatten * self.temperature
                    + phi_sigma * gradeta_pi_at_flatten
            )
            gradeta_flatten = gradeta_flatten + sender_objective_alpha * gradeta_constraint_flatten

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
    def __init__(self):
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=1, bias=False, dtype=torch.double),
            torch.nn.LeakyReLU()  # 算了一下，如果不加activation那么是0个解
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.double),
            torch.nn.Softmax(dim=-1)
        )

        if config_env.initialize:
            set_net_params(self.critic, params=config_env.critic_params)
            set_net_params(self.actor, params=config_env.actor_params)

        self.critic_loss_criterion = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr_actor)

    def build_connection(self, pro):
        self.pro = pro

    def choose_action(self, message):
        pi = self.actor(message)
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

    def update_ac(self, transition, calculate_for_upper_level=True):
        # 目前写的更新是，生成一个trajectory，把里面所有的transition都拿来当成一个batch练; 这个环境下，一个trajectory只有一个transition

        a_onehot_hr = int_to_onehot(transition.a_int_hr)
        message_and_a = torch.cat([transition.message_onehot_pro, a_onehot_hr])

        q = self.critic(message_and_a).squeeze()
        q_next = 0

        td_target = transition.reward_hr + q_next
        td_target = torch.tensor(td_target, dtype=torch.double)  # 没梯度的，没事
        td_error = td_target - q

        critic_loss = self.critic_loss_criterion(td_target, q)  # 没梯度的，没事

        v = self.calculate_v(transition.message_onehot_pro, transition.a_prob_hr)
        advantage = q - v
        # actor_obj = transition.reward_hr * transition.a_logprob_hr
        actor_obj = td_error * transition.a_logprob_hr
        # actor_obj = advantage * transition.a_logprob_hr

        if calculate_for_upper_level:
            '''还要计算upper level的gradetai_Vj; 这里的j指的就是当前这个agent'''
            # print('Calculating gradetai_Vj.')
            self.gradeta_v = torch.autograd.grad(v, list(self.pro.signaling_net.parameters()), retain_graph=True)

            '''还要计算gradthetaj_logpij_list，这是旧的，要在这先算好，在upper level里才能用'''
            # print('Calculating gradthetaj_logpij_list.')
            self.gradtheta_logpi = torch.autograd.grad(transition.a_logprob_hr, list(self.actor.parameters()),
                                                       retain_graph=True, )

            self.lower_level_critic_loss = critic_loss.detach()
            self.lower_level_td_target = td_target.detach()
            # self.lower_level_td_error = td_error.detach()
            self.lower_level_actor_obj = actor_obj.detach()

            raise NotImplementedError('check')

        '''更新'''
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


def set_Env_and_Agents(config):
    print('----------------------------------------')
    print('Setting agents and env.')

    env = recommendation.recommendation_env(config)
    pro = pro_class()
    hr = hr_class()

    pro.build_connection(hr)
    hr.build_connection(pro)

    return env, pro, hr


def run_an_episode(env, pro, hr, ith_episode=None, pls_print=False):
    obs_pro = env.reset().double()  # obs_pro == student_charac
    message_onehot_pro, message_prob_pro, message_pro = pro.send_message(obs_pro, )  # message_pro == letter
    a_int_hr, a_prob_hr, a_logprob_hr = hr.choose_action(message_onehot_pro)  # a_int_hr == hire_decision
    reward_pro, reward_hr = env.step(obs_pro, a_int_hr)

    transition = transition_class()
    transition.set_values([obs_pro,
                           message_onehot_pro, message_prob_pro, message_pro,
                           a_int_hr, a_prob_hr, a_logprob_hr,
                           reward_pro, reward_hr])

    if pls_print:
        print('----------------------------------------')
        if ith_episode or type(ith_episode) is int and ith_episode == 0:
            print('The ' + str(ith_episode) + ' episode finished.')
        print('pro_obs:\t', student_charc_maptoword[obs_pro.detach().int()])
        if message_pro.detach() > 0.5:
            message_idx = 1
        else:
            message_idx = 0
        print('message:\t', professor_action_maptoword[message_idx])
        print('mess_OH:\t', message_onehot_pro.tolist())
        print('pro_prob:\t', message_prob_pro.detach().tolist())
        if a_int_hr.detach() > 0.5:
            action_idx = 1
        else:
            action_idx = 0
        print('hr_action:\t', HR_action_maptoword[action_idx])
        print('hr_prob:\t', a_prob_hr.detach().tolist())
        print('pro_reward:\t', reward_pro)
        print('hr_reward:\t', reward_hr)

    return transition


class transition_class(object):

    def __init__(self, ):
        self.reset()

    def reset(self):
        self.obs_pro, \
        self.message_onehot_pro, self.message_prob_pro, self.message_pro, \
        self.a_int_hr, self.a_prob_hr, self.a_logprob_hr, \
        self.reward_pro, self.reward_hr = [None] * 9

        self.transition = None
        return

    def set_values(self, transition):
        self.obs_pro, \
        self.message_onehot_pro, self.message_prob_pro, self.message_pro, \
        self.a_int_hr, self.a_prob_hr, self.a_logprob_hr, \
        self.reward_pro, self.reward_hr = transition

        self.transition = transition
        return


def train():
    print('----------------------------------------')
    fake_buffer = []
    print('Start training.')
    i_episode = 0
    while i_episode < train_n_episodes:
        transition = run_an_episode(env, pro, hr, ith_episode=i_episode, pls_print=False)
        hr.update_ac(transition, calculate_for_upper_level=False)
        pro.update_c(transition)
        fake_buffer.append(transition.transition)
        i_episode += 1

        if not config_env.fixed_signaling_scheme:
            transition = run_an_episode(env, pro, hr, ith_episode=i_episode, pls_print=False)
            pro.update_infor_design(transition)
            fake_buffer.append(transition.transition)
            i_episode += 1

        if not i_episode % 1e3:
            completion_rate = i_episode / train_n_episodes
            print('Task completion:\t{:.1%}'.format(completion_rate))

    return fake_buffer


def performance_test_and_print(howmany_episodes=50):
    print('----------------------------------------')
    reward_pro = 0
    reward_hr = 0
    social_welfare = 0
    for i in range(howmany_episodes):
        transition = run_an_episode(env, pro, hr, ith_episode=i, pls_print=False)

        i_reward_pro, i_reward_hr = transition.reward_pro, transition.reward_hr
        i_social_welfare = i_reward_pro + i_reward_hr

        reward_pro += i_reward_pro
        reward_hr += i_reward_hr
        social_welfare += i_social_welfare
    reward_pro /= howmany_episodes
    reward_hr /= howmany_episodes
    social_welfare /= howmany_episodes
    print('ave_reward_pro:\t{}'.format(reward_pro))
    print('ave_reward_hr:\t{}'.format(reward_hr))
    print('ave_social_welfare:\t{}'.format(social_welfare))


def print_params():
    print('----------------------------------------')

    signaling_params = [param.tolist() for param in pro.signaling_net.parameters()]
    critic_params = [param.tolist() for param in hr.critic.parameters()]
    actor_params = [param.tolist() for param in hr.actor.parameters()]

    print('signaling_params:', signaling_params)
    print('critic_params:', critic_params)
    print('actor_params:', actor_params)


def unittest_which_weights_are_active():
    # 把actor的sigmoid层注释掉就知道了
    print_params()
    message_onehot_temp1 = torch.tensor([1, 0], dtype=torch.double)
    message_onehot_temp2 = torch.tensor([0, 1], dtype=torch.double)
    actor_parameters = list(hr.actor.parameters())
    actor_parameters[0].data = torch.tensor([[1, 2], [3, 4]], dtype=torch.double)
    print_params()
    _, pi1, _ = hr.choose_action(message_onehot_temp1)
    _, pi2, _ = hr.choose_action(message_onehot_temp2)
    print(pi1.tolist())
    print(pi2.tolist())


def permutation_test():
    print('----------------------------------------')
    for student_quality in range(2):
        _, phi_current, _ = pro.send_message(student_quality)
        print('phi(message|s={})=\t'.format(student_quality), phi_current.tolist())
    for message_int in range(2):
        message_onehot = [0, 0]
        message_onehot[message_int] = 1
        message_onehot = torch.tensor(message_onehot, dtype=torch.double)
        _, pi, _ = hr.choose_action(message_onehot)
        print('pi(a|message={})=\t'.format(message_int), pi.tolist())


def plot_all(fake_buffer):
    fig = plt.figure()  # 画布
    fig.canvas.manager.set_window_title('Recommendation Letter Env.')

    reward_pro_curve = fig.add_subplot(4, 2, 1)
    reward_hr_curve = fig.add_subplot(4, 2, 3)

    reward_compare_curve = fig.add_subplot(4, 2, 2)
    reward_socialwalfare_curve = fig.add_subplot(4, 2, 4)

    phi_rec_when_bad_curve = fig.add_subplot(4, 2, 5)
    phi_rec_when_good_curve = fig.add_subplot(4, 2, 7)

    pi_hire_when_notrec_curve = fig.add_subplot(4, 2, 6)
    pi_hire_when_rec_curve = fig.add_subplot(4, 2, 8)

    data = np.array(fake_buffer)

    y_reward_pro = data[:, -2]
    y_reward_hr = data[:, -1]

    y_social_welfare = y_reward_pro + y_reward_hr

    x_phi_rec_when_bad = []
    x_phi_rec_when_good = []
    y_phi_rec_when_bad = []
    y_phi_rec_when_good = []
    for i in range(len(data)):
        if data[i][0] < 0.5:  # obs_pro
            x_phi_rec_when_bad.append(i)
            y_phi_rec_when_bad.append(data[i][2][1])  # message_prob_pro
        else:
            x_phi_rec_when_good.append(i)
            y_phi_rec_when_good.append(data[i][2][1])  # message_prob_pro
    y_phi_rec_when_bad = np.array(y_phi_rec_when_bad)
    y_phi_rec_when_good = np.array(y_phi_rec_when_good)

    x_pi_hire_when_notrec = []
    x_pi_hire_when_rec = []
    y_pi_hire_when_notrec = []
    y_pi_hire_when_rec = []
    for i in range(len(data)):
        if data[i][3] < 0.5:  # message_pro
            x_pi_hire_when_notrec.append(i)
            y_pi_hire_when_notrec.append(data[i][-4][1])  # a_prob_hr
        else:
            x_pi_hire_when_rec.append(i)
            y_pi_hire_when_rec.append(data[i][-4][1])  # a_prob_hr
    y_pi_hire_when_notrec = np.array(y_pi_hire_when_notrec)
    y_pi_hire_when_rec = np.array(y_pi_hire_when_rec)

    # 光滑，求平均
    window_size = 101
    polynomial_order = 3
    y_reward_pro = savgol_filter(y_reward_pro, window_size, polynomial_order)
    y_reward_hr = savgol_filter(y_reward_hr, window_size, 3)
    y_social_welfare = savgol_filter(y_social_welfare, window_size, polynomial_order)
    y_phi_rec_when_bad = savgol_filter(y_phi_rec_when_bad, window_size, polynomial_order)
    y_phi_rec_when_good = savgol_filter(y_phi_rec_when_good, window_size, polynomial_order)
    y_pi_hire_when_notrec = savgol_filter(y_pi_hire_when_notrec, window_size, polynomial_order)
    y_pi_hire_when_rec = savgol_filter(y_pi_hire_when_rec, window_size, polynomial_order)

    # 插值，画曲线
    x = list(range(len(data)))
    x_smooth = np.linspace(0, len(data), 100)
    y_smooth_reward_pro = make_interp_spline(x, y_reward_pro)(x_smooth)
    y_smooth_reward_hr = make_interp_spline(x, y_reward_hr)(x_smooth)
    y_smooth_social_welfare = make_interp_spline(x, y_social_welfare)(x_smooth)

    x_smooth_phi_rec_when_bad = np.linspace(0, len(data), len(x_phi_rec_when_bad))
    x_smooth_phi_rec_when_good = np.linspace(0, len(data), len(x_phi_rec_when_good))
    x_smooth_pi_hire_when_notrec = np.linspace(0, len(data), len(x_pi_hire_when_notrec))
    x_smooth_pi_hire_when_rec = np.linspace(0, len(data), len(x_pi_hire_when_rec))

    y_smooth_phi_rec_when_bad = make_interp_spline(x_phi_rec_when_bad, y_phi_rec_when_bad)(x_smooth_phi_rec_when_bad)
    y_smooth_phi_rec_when_good = make_interp_spline(x_phi_rec_when_good, y_phi_rec_when_good)(
        x_smooth_phi_rec_when_good)
    y_smooth_pi_hire_when_notrec = make_interp_spline(x_pi_hire_when_notrec, y_pi_hire_when_notrec)(
        x_smooth_pi_hire_when_notrec)
    y_smooth_pi_hire_when_rec = make_interp_spline(x_pi_hire_when_rec, y_pi_hire_when_rec)(x_smooth_pi_hire_when_rec)

    # 有些时候插值会在边界超出范围,我clip一下
    # y_smooth_reward_pro = np.clip(y_smooth_reward_pro, 0, 1)
    # y_smooth_reward_hr = np.clip(y_smooth_reward_hr, 0, 1)
    # y_smooth_social_welfare = np.clip(y_smooth_social_welfare, 0, 1)
    y_smooth_phi_rec_when_bad = np.clip(y_smooth_phi_rec_when_bad, 0, 1)
    y_smooth_phi_rec_when_good = np.clip(y_smooth_phi_rec_when_good, 0, 1)
    y_smooth_pi_hire_when_notrec = np.clip(y_smooth_pi_hire_when_notrec, 0, 1)
    y_smooth_pi_hire_when_rec = np.clip(y_smooth_pi_hire_when_rec, 0, 1)

    reward_pro_curve.plot(x_smooth, y_smooth_reward_pro)
    reward_hr_curve.plot(x_smooth, y_smooth_reward_hr)

    reward_compare_curve.plot(x_smooth, y_smooth_reward_pro)
    reward_compare_curve.plot(x_smooth, y_smooth_reward_hr)
    reward_socialwalfare_curve.plot(x_smooth, y_smooth_social_welfare)

    phi_rec_when_bad_curve.plot(x_smooth_phi_rec_when_bad, y_smooth_phi_rec_when_bad)
    phi_rec_when_good_curve.plot(x_smooth_phi_rec_when_good, y_smooth_phi_rec_when_good)

    pi_hire_when_notrec_curve.plot(x_smooth_pi_hire_when_notrec, y_smooth_pi_hire_when_notrec)
    pi_hire_when_rec_curve.plot(x_smooth_pi_hire_when_rec, y_smooth_pi_hire_when_rec)
    # 画一起去看看趋势
    pi_hire_when_notrec_curve.plot(x_smooth_phi_rec_when_bad, 1 - y_smooth_phi_rec_when_bad)
    pi_hire_when_rec_curve.plot(x_smooth_phi_rec_when_good, y_smooth_phi_rec_when_good)

    # 设置标题
    reward_pro_curve.set_title('Reward of Professor')
    reward_hr_curve.set_title('Reward of HR')
    reward_compare_curve.set_title('Reward Comparation')
    reward_socialwalfare_curve.set_title('Social Welfare')
    phi_rec_when_bad_curve.set_title('P( signal=1 | bad stu. )')
    phi_rec_when_good_curve.set_title('P( signal=1 | good stu. )')
    pi_hire_when_notrec_curve.set_title('P( hire | signal=0 )')
    pi_hire_when_rec_curve.set_title('P( hire | signal=1 )')

    # 为重合的3张图设置标注legent
    reward_compare_curve.legend(['R of Pro.', 'R of HR'])
    pi_hire_when_notrec_curve.legend(['P(hire|signal=0)', 'P(signal=0|bad)'])
    pi_hire_when_rec_curve.legend(['P(hire|signal=1)', 'P(signal=1|good)'])

    # 设置坐标轴范围
    ylim_min = -0.2
    ylim_max = 1.2
    phi_rec_when_bad_curve.set_ylim(ylim_min, ylim_max)
    phi_rec_when_good_curve.set_ylim(ylim_min, ylim_max)
    pi_hire_when_notrec_curve.set_ylim(ylim_min, ylim_max)
    pi_hire_when_rec_curve.set_ylim(ylim_min, ylim_max)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    env, pro, hr = set_Env_and_Agents(config_env)

    # print_params()
    fake_buffer = train()
    plot_all(fake_buffer)
    # print_params()

    print('----------------------------------------')
    print('All done.')
