import torch
import numpy as np
from exp_recommendation.episode_generator import run_an_episode

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import wandb
from exp_recommendation.mykey import wandb_login_key, wandb_project_name, wandb_entity_name


def set_net_params(net, params):
    net_params_list = list(net.parameters())
    for i in range(len(net_params_list)):
        net_params_list[i].data = params[i].clone()
    return


def int_to_onehot(variable_int_list, k, device):
    variable_onehot = torch.zeros(len(variable_int_list), k, dtype=torch.float64, device=device)
    variable_onehot[range(len(variable_int_list)), variable_int_list] = 1
    return variable_onehot


def performance_test_and_print(howmany_episodes, env, pro, hr):
    print('----------------------------------------')
    reward_pro = 0
    reward_hr = 0
    social_welfare = 0
    for i in range(howmany_episodes):
        transition = run_an_episode(env, pro, hr)

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
    critic_pro_forhr_params = [param.tolist() for param in pro.critic_forhr.parameters()]
    signaling_params = [param.tolist() for param in pro.signaling_net.parameters()]
    critic_params = [param.tolist() for param in hr.critic.parameters()]
    actor_params = [param.tolist() for param in hr.actor.parameters()]

    print('critic_pro_params:', critic_pro_params)
    print('critic_pro_forhr_params:', critic_pro_forhr_params)
    print('signaling_params:', signaling_params)
    print('critic_params:', critic_params)
    print('actor_params:', actor_params)


def validate_critic(agent, device):
    print('----------------------------------------')
    for input1 in range(2):
        for input2 in range(2):
            input1_onehot = [0, 0]
            input2_onehot = [0, 0]
            input1_onehot[input1] = 1
            input2_onehot[input2] = 1
            input = torch.tensor(input1_onehot + input2_onehot, dtype=torch.float64, device=device)
            q = agent.critic(input)
            if agent.name == 'pro':
                Gj = agent.critic_forhr(input)
                print('state:{}, action:{}, Gi:{}'.format(input1, input2, q))
                print('state:{}, action:{}, Gj:{}'.format(input1, input2, Gj))
            elif agent.name == 'hr':
                print('signal:{}, action:{}, q:{}'.format(input1, input2, q))
    return


def validate_sig_actor(agent, device):
    print('----------------------------------------')
    for input in range(2):
        if agent.name == 'pro':
            input_tensor = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(dim=0)
            _, phi, _ = agent.send_message(input_tensor)
            print('state:{}, phi:{}'.format(input, phi))
        elif agent.name == 'hr':
            input_onehot = [0, 0]
            input_onehot[input] = 1
            input_tensor = torch.tensor(input_onehot, dtype=torch.float64, device=device)
            _, pi, _ = agent.choose_action(input_tensor)
            print('signal:{}, pi:{}'.format(input, pi))
        else:
            raise NotImplementedError

    return


def validate(pro, hr, device):
    validate_critic(pro, device)
    validate_sig_actor(pro, device)
    print('----------------------------------------')
    validate_critic(hr, device)
    validate_sig_actor(hr, device)


def unittest_which_weights_are_active(pro, hr):
    # 把actor的sigmoid层注释掉就知道了
    print_params(pro, hr)
    message_onehot_temp1 = torch.tensor([1, 0], dtype=torch.float64)
    message_onehot_temp2 = torch.tensor([0, 1], dtype=torch.float64)
    actor_parameters = list(hr.actor.parameters())
    actor_parameters[0].data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
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
        message_onehot = torch.tensor(message_onehot, dtype=torch.float64)
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
    np.random.seed(myseed)  # numpy seed
    torch.manual_seed(myseed)  # torch seed
    torch.cuda.manual_seed(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_create_canvas():
    fig = plt.figure(figsize=(8, 6), dpi=300)  # canvas
    fig.canvas.manager.set_window_title('Recommendation Letter Env.')

    reward_pro_curve = fig.add_subplot(4, 2, 1)
    reward_hr_curve = fig.add_subplot(4, 2, 3)

    reward_compare_curve = fig.add_subplot(4, 2, 2)
    reward_socialwalfare_curve = fig.add_subplot(4, 2, 4)

    phi_rec_when_bad_curve = fig.add_subplot(4, 2, 5)
    phi_rec_when_good_curve = fig.add_subplot(4, 2, 7)

    pi_hire_when_notrec_curve = fig.add_subplot(4, 2, 6)
    pi_hire_when_rec_curve = fig.add_subplot(4, 2, 8)

    # title
    reward_pro_curve.set_title('Reward of Professor')
    reward_hr_curve.set_title('Reward of HR')
    reward_compare_curve.set_title('Reward Comparation')
    reward_socialwalfare_curve.set_title('Social Welfare')
    phi_rec_when_bad_curve.set_title('P( signal=1 | bad stu. )')
    phi_rec_when_good_curve.set_title('P( signal=1 | good stu. )')
    pi_hire_when_notrec_curve.set_title('P( hire | signal=0 )')
    pi_hire_when_rec_curve.set_title('P( hire | signal=1 )')

    # scientific notation
    for axis in fig.axes:
        axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    # legends
    # reward_compare_curve.legend(['R of Pro.', 'R of HR'])
    # pi_hire_when_notrec_curve.legend(['P(hire|signal=0)', 'P(signal=0|bad)'])
    # pi_hire_when_rec_curve.legend(['P(hire|signal=1)', 'P(signal=1|good)'])

    # axes limits
    ylim_min = -0.2
    ylim_max = 1.2
    phi_rec_when_bad_curve.set_ylim(ylim_min, ylim_max)
    phi_rec_when_good_curve.set_ylim(ylim_min, ylim_max)
    pi_hire_when_notrec_curve.set_ylim(ylim_min, ylim_max)
    pi_hire_when_rec_curve.set_ylim(ylim_min, ylim_max)

    plt.tight_layout()

    return fig, \
           reward_pro_curve, reward_hr_curve, \
           reward_compare_curve, reward_socialwalfare_curve, \
           phi_rec_when_bad_curve, phi_rec_when_good_curve, \
           pi_hire_when_notrec_curve, pi_hire_when_rec_curve


def plot_all(fake_buffer, fig, reward_pro_curve, reward_hr_curve,
             reward_compare_curve, reward_socialwalfare_curve,
             phi_rec_when_bad_curve, phi_rec_when_good_curve,
             pi_hire_when_notrec_curve, pi_hire_when_rec_curve):
    data = np.array(fake_buffer, dtype=object)

    y_reward_pro = data[-2]
    y_reward_hr = data[-1]

    y_social_welfare = y_reward_pro + y_reward_hr

    x_phi_rec_when_bad = (data[0] == 0).nonzero(as_tuple=False).squeeze(dim=1)  # student == 0
    x_phi_rec_when_good = (data[0] == 1).nonzero(as_tuple=False).squeeze(dim=1)
    y_phi_rec_when_bad = data[2][x_phi_rec_when_bad, 1]  # prob of rec
    y_phi_rec_when_good = data[2][x_phi_rec_when_good, 1]

    x_pi_hire_when_notrec = (data[3] < 0.5).nonzero(as_tuple=False).squeeze(dim=1)  # message_pro == 0
    x_pi_hire_when_rec = (data[3] > 0.5).nonzero(as_tuple=False).squeeze(dim=1)
    y_pi_hire_when_notrec = data[-4][x_pi_hire_when_notrec, 1]  # prob of hire
    y_pi_hire_when_rec = data[-4][x_pi_hire_when_rec, 1]

    # smoothing
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
    # pi_hire_when_notrec_curve.plot(x_smooth_phi_rec_when_bad, 1 - y_smooth_phi_rec_when_bad)
    # pi_hire_when_rec_curve.plot(x_smooth_phi_rec_when_good, y_smooth_phi_rec_when_good)


def init_wandb(group_name, sender_objective_alpha):
    # wandb.login(key=wandb_login_key)
    run_handle = wandb.init(project=wandb_project_name, entity=wandb_entity_name, reinit=True, group=group_name,
                            config={"pro_objective_alpha": sender_objective_alpha})

    chart_name_list = ['reward_sender',
                       'reward_receiver',
                       'social_welfare',

                       'prob_of_signaling_1_when_bad',
                       'prob_of_signaling_1_when_good',
                       'prob_of_hiring_when_0',
                       'prob_of_hiring_when_1',
                       ]

    for chart_name in chart_name_list:
        wandb.define_metric(chart_name)

    return chart_name_list, run_handle


def plot_with_wandb(chart_name_list, batch, i_episode, env_sample_n_students):
    entry = dict(zip(chart_name_list, [0] * len(chart_name_list)))

    state = batch.obs_pro.detach()
    message_pro = batch.message_pro.detach()

    reward_pro_tensor = batch.reward_pro.detach()
    reward_hr_tensor = batch.reward_hr.detach()
    reward_tot_tensor = reward_pro_tensor + reward_hr_tensor

    message_prob_pro = batch.message_prob_pro.detach()
    a_prob_hr = batch.a_prob_hr.detach()

    entry['reward_sender'] = float(torch.mean(reward_pro_tensor))
    entry['reward_receiver'] = float(torch.mean(reward_hr_tensor))
    entry['social_welfare'] = float(torch.mean(reward_tot_tensor))

    idx_bad_stu = (state == 0).nonzero().squeeze()
    idx_good_stu = (state == 1).nonzero().squeeze()
    phi_when_bad = torch.index_select(message_prob_pro, 0, idx_bad_stu)
    phi_when_good = torch.index_select(message_prob_pro, 0, idx_good_stu)
    entry['prob_of_signaling_1_when_bad'] = float(torch.mean(phi_when_bad, dim=0)[1])
    entry['prob_of_signaling_1_when_good'] = float(torch.mean(phi_when_good, dim=0)[1])

    idx_message_0 = (message_pro == 0).nonzero().squeeze()
    idx_message_1 = (message_pro == 1).nonzero().squeeze()
    pi_when_0 = torch.index_select(a_prob_hr, 0, idx_message_0)
    pi_when_1 = torch.index_select(a_prob_hr, 0, idx_message_1)
    entry['prob_of_hiring_when_0'] = float(torch.mean(pi_when_0, dim=0)[1])
    entry['prob_of_hiring_when_1'] = float(torch.mean(pi_when_1, dim=0)[1])

    wandb.log(entry, step=i_episode * env_sample_n_students)

    # for idx in range(len(state)):
    #     entry['reward_sender'] = float(reward_pro_tensor[idx])
    #     entry['reward_receiver'] = float(reward_hr_tensor[idx])
    #     entry['social_welfare'] = float(reward_tot_tensor[idx])
    #
    #     if state[idx] == 0:
    #         entry['prob_of_signaling_1_when_bad'] = float(message_prob_pro[idx][1])
    #     else:
    #         entry['prob_of_signaling_1_when_good'] = float(message_prob_pro[idx][1])
    #
    #     if message_pro[idx] == 0:
    #         entry['prob_of_hiring_when_0'] = float(a_prob_hr[idx][1])
    #     else:
    #         entry['prob_of_hiring_when_1'] = float(a_prob_hr[idx][1])
    #     wandb.log(entry, step=idx + i_episode)
    return
