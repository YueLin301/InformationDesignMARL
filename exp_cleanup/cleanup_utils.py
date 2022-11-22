import torch
import numpy as np
from exp_recommendation.episode_generator import run_an_episode

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline  # 曲线插值
from scipy.signal import savgol_filter


def set_net_params(net, params):
    net_params_list = list(net.parameters())
    for i in range(len(net_params_list)):
        net_params_list[i].data = params[i].clone()
    return


def int_to_onehot(variable_int):
    variable_onehot = [0, 0]
    variable_onehot[variable_int] = 1
    variable_onehot = torch.tensor(variable_onehot, dtype=torch.double)
    return variable_onehot


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


def validate_critic(agent):
    print('----------------------------------------')
    for input1 in range(2):
        for input2 in range(2):
            input1_onehot = [0, 0]
            input2_onehot = [0, 0]
            input1_onehot[input1] = 1
            input2_onehot[input2] = 1
            input = torch.tensor(input1_onehot + input2_onehot, dtype=torch.double)
            q = agent.critic(input)
            if agent.name == 'pro':
                print('state:{}, signal:{}, q:{}'.format(input1, input2, q))
            elif agent.name == 'hr':
                print('signal:{}, action:{}, q:{}'.format(input1, input2, q))
    return


def validate_sig_actor(agent):
    print('----------------------------------------')
    for input in range(2):
        if agent.name == 'pro':
            input_tensor = torch.tensor(input, dtype=torch.double)
            _, phi, _ = agent.send_message(input_tensor)
            print('state:{}, phi:{}'.format(input, phi))
        elif agent.name == 'hr':
            input_onehot = [0, 0]
            input_onehot[input] = 1
            input_tensor = torch.tensor(input_onehot, dtype=torch.double)
            _, pi, _ = agent.choose_action(input_tensor)
            print('signal:{}, pi:{}'.format(input, pi))
        else:
            raise NotImplementedError

    return


def validate(pro, hr):
    validate_critic(pro)
    validate_sig_actor(pro)
    print('----------------------------------------')
    validate_critic(hr)
    validate_sig_actor(hr)


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


def plot_create_canvas():
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
    # reward_compare_curve.legend(['R of Pro.', 'R of HR'])
    # pi_hire_when_notrec_curve.legend(['P(hire|signal=0)', 'P(signal=0|bad)'])
    # pi_hire_when_rec_curve.legend(['P(hire|signal=1)', 'P(signal=1|good)'])

    # 设置坐标轴范围
    ylim_min = -0.2
    ylim_max = 1.2
    phi_rec_when_bad_curve.set_ylim(ylim_min, ylim_max)
    phi_rec_when_good_curve.set_ylim(ylim_min, ylim_max)
    pi_hire_when_notrec_curve.set_ylim(ylim_min, ylim_max)
    pi_hire_when_rec_curve.set_ylim(ylim_min, ylim_max)

    plt.tight_layout()

    return reward_pro_curve, reward_hr_curve, \
           reward_compare_curve, reward_socialwalfare_curve, \
           phi_rec_when_bad_curve, phi_rec_when_good_curve, \
           pi_hire_when_notrec_curve, pi_hire_when_rec_curve


def plot_all(fake_buffer, reward_pro_curve, reward_hr_curve,
             reward_compare_curve, reward_socialwalfare_curve,
             phi_rec_when_bad_curve, phi_rec_when_good_curve,
             pi_hire_when_notrec_curve, pi_hire_when_rec_curve):
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
    window_size = 501
    polynomial_order = 3
    y_reward_pro = savgol_filter(y_reward_pro, window_size, polynomial_order)
    y_reward_hr = savgol_filter(y_reward_hr, window_size, 3)
    y_social_welfare = savgol_filter(y_social_welfare, window_size, polynomial_order)
    window_size2 = 41
    polynomial_order2 = 2
    y_phi_rec_when_bad = savgol_filter(y_phi_rec_when_bad, window_size2, polynomial_order2)
    y_phi_rec_when_good = savgol_filter(y_phi_rec_when_good, window_size2, polynomial_order2)
    y_pi_hire_when_notrec = savgol_filter(y_pi_hire_when_notrec, window_size2, polynomial_order2)
    y_pi_hire_when_rec = savgol_filter(y_pi_hire_when_rec, window_size2, polynomial_order2)

    reward_pro_curve.plot(y_reward_pro)
    reward_hr_curve.plot(y_reward_hr)

    reward_compare_curve.plot(y_reward_pro)
    reward_compare_curve.plot(y_reward_hr)
    reward_socialwalfare_curve.plot(y_social_welfare)

    phi_rec_when_bad_curve.plot(y_phi_rec_when_bad)
    phi_rec_when_good_curve.plot(y_phi_rec_when_good)

    pi_hire_when_notrec_curve.plot(y_pi_hire_when_notrec)
    pi_hire_when_rec_curve.plot(y_pi_hire_when_rec)
    # 画一起去看看趋势
    # pi_hire_when_notrec_curve.plot(x_smooth_phi_rec_when_bad, 1 - y_smooth_phi_rec_when_bad)
    # pi_hire_when_rec_curve.plot(x_smooth_phi_rec_when_good, y_smooth_phi_rec_when_good)
