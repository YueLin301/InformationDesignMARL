from env import cleanup

from exp_cleanup.transition_class import transition_class
from exp_cleanup.configs.configfile_acb_learn_equalcol import config
from exp_cleanup.agent_class import sender_class, receiver_class

sender_id = 0  # 0 or 1


def set_Env_and_Agents(config):
    env = cleanup.Env(config.env)

    sender = sender_class(config)
    receiver = receiver_class(config)

    sender.build_connection(receiver)
    receiver.build_connection(sender)

    return env, sender, receiver


def run_an_episode(env, sender, receiver):
    obs_list_np = env.reset()
    obs_sender = obs_list_np[sender_id]  # obs of receiver is not used
    done = False

    while not done:
        message, message_prob = sender.send_message(obs_sender)

        a_int_sender, a_prob_sender, a_logprob_sender = sender.choose_action(obs_sender, using_epsilon=True)
        a_int_receiver, a_prob_receiver, a_logprob_receiver = receiver.choose_action(message, using_epsilon=True)

        obs_next_list_np, rewards_list, done, info = env.step(obs_sender, [a_int_sender, a_int_receiver])
        obs_next_sender = obs_next_list_np[sender_id]
        reward_sender, reward_receiver = rewards_list

        transition = transition_class()
        transition.set_values([obs_sender,
                               message, message_prob,
                               a_int_sender, a_prob_sender, a_logprob_sender,
                               a_int_receiver, a_prob_receiver, a_logprob_receiver,
                               reward_sender, reward_receiver])

        if not done:
            obs_sender = obs_next_sender

    return
