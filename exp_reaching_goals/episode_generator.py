import torch
import os.path
import imageio


def run_an_episode(env, sender, receiver, config, device, pls_render, buffer):
    done = False
    state = env.reset().to(device)  # receiver position, sender apple position, receiver apple position
    step = 0
    pic_step = 0  # before/after act

    while not done:
        obs_sender = state if not config.sender.regradless_agent_pos else state[:, 1:, :, :]
        if not config.sender.honest:
            message, phi = sender.send_message(obs_sender)
            obs_range = config.receiver.obs_range
            obs_and_message_receiver = message
            if obs_range[1]:
                obs_and_message_receiver = torch.cat([state[:, 2:3, :, :], obs_and_message_receiver], dim=1)
            if obs_range[0]:
                obs_and_message_receiver = torch.cat([state[:, 0:1, :, :], obs_and_message_receiver], dim=1)
        else:
            obs_and_message_receiver, message, phi = obs_sender, None, None
            if not config.env.aligned_object and config.n_channels.obs_and_message_receiver == 2:
                obs_and_message_receiver = torch.cat([obs_and_message_receiver[:, 0:1, :, :],
                                                      obs_and_message_receiver[:, 2:3, :, :]], dim=1)
            else:
                raise NotImplementedError()

        if pls_render:
            filename = os.path.join(config.path.saved_episode, str(pic_step))
            env.render(step=step, type='before', message=message, phi=phi,
                       pi=torch.zeros(config.env.dim_action).unsqueeze(dim=0), filename=filename)
            pic_step += 1

        a_receiver, pi = receiver.choose_action(obs_and_message_receiver)
        state_next, rewards, done = env.step(int(a_receiver))

        reward_sender = torch.tensor(rewards[sender.id], dtype=torch.double).to(device).unsqueeze(
            dim=0) if not config.sender.honest else None
        reward_receiver = torch.tensor(rewards[receiver.id], dtype=torch.double).to(device).unsqueeze(dim=0)

        half_transition = [obs_sender.to(device), message, phi, obs_and_message_receiver, a_receiver, pi,
                           reward_sender, reward_receiver]
        half_transition_clone = [half_transition[i].clone() if not half_transition[i] is None else None for i in
                                 range(len(half_transition))]

        if not done:  # the last transition
            buffer.add_half_transition(half_transition, '1st')
        else:
            buffer.add_half_transition(half_transition, '1st')
            half_transition_clone[-1] = torch.zeros(1).to(device)
            half_transition_clone[-2] = torch.zeros(1).to(device)
            buffer.add_half_transition(half_transition_clone, '2nd')
        if step:  # the first transition
            buffer.add_half_transition(half_transition_clone, '2nd')

        if pls_render:
            filename = os.path.join(config.path.saved_episode, str(pic_step))
            env.render(step=step, type='after', message=message, phi=phi, pi=pi, filename=filename)
            pic_step += 1

        step += 1
        state = state_next.to(device)

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
