import torch
from exp_harvest.buffer_class import buffer_class


def run_an_episode(env, pro, hr):
    obs_pro = env.reset()
    message_onehot_pro, message_prob_pro, message_pro = pro.send_message(obs_pro)
    a_int_hr, a_prob_hr, a_logprob_hr = hr.choose_action(message_onehot_pro)
    reward_pro, reward_hr = env.step(obs_pro, a_int_hr)

    buffer = buffer_class()
    buffer.add([obs_pro,
                message_onehot_pro, message_prob_pro, message_pro,
                a_int_hr, a_prob_hr, a_logprob_hr,
                reward_pro, reward_hr])

    return buffer
