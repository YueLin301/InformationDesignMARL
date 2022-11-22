from exp_recommendation.transition_class import transition_class
from env.recommendation import student_charc_maptoword, professor_action_maptoword, HR_action_maptoword


def run_an_episode(env, sender, receiver, ith_episode=None, pls_print=False):
    obs_np = env.reset()
    obs_sender = obs_np[0]
    done = False

    while not done:
        message, message_prob = sender.send_message(obs_sender)

        a_int_sender, a_prob_sender, a_logprob_sender = sender.choose_action(obs_sender, using_epsilon=True)
        a_int_receiver, a_prob_receiver, a_logprob_receiver = receiver.choose_action(message, using_epsilon=True)

        reward_sender, reward_receiver = env.step(obs_sender, [a_int_sender, a_int_receiver])

        transition = transition_class()
        transition.set_values([obs_sender,
                               message, message_prob,
                               a_int_sender, a_prob_sender, a_logprob_sender,
                               a_int_receiver, a_prob_receiver, a_logprob_receiver,
                               reward_sender, reward_receiver])

    return
