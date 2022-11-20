import torch
import numpy as np


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


from exp_recommendation.episode_generator import run_an_episode


def performance_test_and_print(howmany_episodes, env, pro, hr):
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


def print_params(pro, hr):
    print('----------------------------------------')

    critic_pro_params = [param.tolist() for param in pro.critic.parameters()]
    signaling_params = [param.tolist() for param in pro.signaling_net.parameters()]
    critic_params = [param.tolist() for param in hr.critic.parameters()]
    actor_params = [param.tolist() for param in hr.actor.parameters()]

    print('critic_pro_params:', critic_pro_params)
    print('signaling_params:', signaling_params)
    print('critic_params:', critic_params)
    print('actor_params:', actor_params)


def unittest_which_weights_are_active(pro, hr):
    # 把actor的sigmoid层注释掉就知道了
    print_params(pro, hr)
    message_onehot_temp1 = torch.tensor([1, 0], dtype=torch.double)
    message_onehot_temp2 = torch.tensor([0, 1], dtype=torch.double)
    actor_parameters = list(hr.actor.parameters())
    actor_parameters[0].data = torch.tensor([[1, 2], [3, 4]], dtype=torch.double)
    print_params(pro, hr)
    _, pi1, _ = hr.choose_action(message_onehot_temp1)
    _, pi2, _ = hr.choose_action(message_onehot_temp2)
    print(pi1.tolist())
    print(pi2.tolist())


def permutation_test(pro, hr):
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


def flatten_layers(gradtensor, dim=0):
    gradtensor_flatten = torch.flatten(gradtensor[0])
    for layerl in range(1, len(gradtensor)):
        temp = torch.flatten(gradtensor[layerl])
        gradtensor_flatten = torch.cat([gradtensor_flatten, temp])
    gradtensor_flatten = gradtensor_flatten.unsqueeze(dim=dim)

    return gradtensor_flatten


def set_seed(myseed):
    np.random.seed(myseed)  # numpy的随机种子
    torch.manual_seed(myseed)  # torch的随机种子
    torch.cuda.manual_seed(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
