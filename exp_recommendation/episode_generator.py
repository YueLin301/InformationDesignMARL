import torch
from exp_recommendation.buffer_class import buffer_class
from env.recommendation import student_charc_maptoword, professor_action_maptoword, HR_action_maptoword


def run_an_episode(env, pro, hr, fake_buffer=None):
    obs_pro = env.reset()
    message_onehot_pro, message_prob_pro, message_pro = pro.send_message(obs_pro)
    a_int_hr, a_prob_hr, a_logprob_hr = hr.choose_action(message_onehot_pro)
    reward_pro, reward_hr = env.step(obs_pro, a_int_hr)

    buffer = buffer_class()
    buffer.set_values([obs_pro,
                       message_onehot_pro, message_prob_pro, message_pro,
                       a_int_hr, a_prob_hr, a_logprob_hr,
                       reward_pro, reward_hr])

    # torch.cat for fake_buffer
    if not len(fake_buffer):
        fake_buffer = buffer.values
        for idx in range(len(fake_buffer)):
            fake_buffer[idx] = fake_buffer[idx].detach()
    else:
        for idx in range(len(fake_buffer)):
            fake_buffer[idx] = torch.cat([fake_buffer[idx], buffer.values[idx].detach()])

    return buffer, fake_buffer
