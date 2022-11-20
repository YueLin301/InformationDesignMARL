from exp_recommendation.transition_class import transition_class
from env.recommendation import student_charc_maptoword, professor_action_maptoword, HR_action_maptoword


def run_an_episode(env, pro, hr, ith_episode=None, pls_print=False):
    obs_pro = env.reset().double()  # obs_pro == student_charac
    message_onehot_pro, message_prob_pro, message_pro = pro.send_message(obs_pro, )  # message_pro == letter
    a_int_hr, a_prob_hr, a_logprob_hr = hr.choose_action(message_onehot_pro,
                                                         using_epsilon=True)  # a_int_hr == hire_decision
    reward_pro, reward_hr = env.step(obs_pro, a_int_hr)

    transition = transition_class()
    transition.set_values([obs_pro,
                           message_onehot_pro, message_prob_pro, message_pro,
                           a_int_hr, a_prob_hr, a_logprob_hr,
                           reward_pro, reward_hr])

    hr.epsilon = hr.epsilon * hr.config.hr.epsilon_decay

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
