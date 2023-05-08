import torch
import os.path
import imageio


def run_an_episode(env, sender, receiver_list, config, pls_render, buffer):
    done = False
    state = env.reset().to(torch.double)  # receiver position, sender apple position, receiver apple position
    step = 0
    pic_step = 0  # before/after act

    nj = env.nj

    while not done:
        message, phi = sender.send_message(state)

        obs_range = config.receiver.obs_range
        obs_and_message_receiver = message.unsqueeze(dim=2)
        if obs_range[1]:
            obs_and_message_receiver = torch.cat([state[:, 1:nj + 1, :, :].unsqueeze(dim=2),
                                                  obs_and_message_receiver], dim=2)
        if obs_range[0]:
            obs_and_message_receiver = torch.cat([state[:, nj + 1:2 * nj + 1, :, :].unsqueeze(dim=2),
                                                  obs_and_message_receiver], dim=2)

        if pls_render:
            filename = os.path.join(config.path.saved_episode, str(pic_step))
            env.render(step=step, type='before', filename=filename)
            pic_step += 1

        a_receiver_list, pi_list, a_receiver_listint = [], [], []
        for j in range(nj):
            a_receiver, pi = receiver_list[j].choose_action(obs_and_message_receiver[:, j, :, :, :])
            a_receiver_list.append(a_receiver)
            a_receiver_listint.append(int(a_receiver))
            pi_list.append(pi)
        a_receiver_tensor, pi_tensor = torch.cat(a_receiver_list).unsqueeze(dim=0), torch.cat(pi_list).unsqueeze(dim=0)

        state_next, i_reward, j_rewards, done = env.step(a_receiver_listint)

        half_transition = [state, message, phi, obs_and_message_receiver, a_receiver_tensor, pi_tensor,
                           i_reward.unsqueeze(dim=0), j_rewards.unsqueeze(dim=0)]
        half_transition_clone = [half_transition[i].clone() if not half_transition[i] is None else None for i in
                                 range(len(half_transition))]

        if not done:  # the last transition
            buffer.add_half_transition(half_transition, '1st')
        else:
            buffer.add_half_transition(half_transition, '1st')
            half_transition_clone[-1] = torch.zeros(1, nj)
            half_transition_clone[-2] = torch.zeros(1)
            buffer.add_half_transition(half_transition_clone, '2nd')
        if step:  # the first transition
            buffer.add_half_transition(half_transition_clone, '2nd')

        if pls_render:
            filename = os.path.join(config.path.saved_episode, str(pic_step))
            env.render(step=step, type='after', filename=filename)
            pic_step += 1

        step += 1
        state = state_next.to(torch.double)

    if pls_render:
        imgs = []
        for file_idx in range(pic_step):
            filename = os.path.join(config.path.saved_episode, str(file_idx) + '.png')
            # img = imageio.v2.imread(filename)
            img = imageio.imread(filename)
            imgs.append(img)
        imageio.mimsave(os.path.join(config.path.saved_episode, 'generated_episode.gif'), imgs, duration=0.25)

    if not sender is None and sender.epsilon > config.sender.epsilon_min:
        sender.epsilon = sender.epsilon * config.sender.epsilon_decay
    return
