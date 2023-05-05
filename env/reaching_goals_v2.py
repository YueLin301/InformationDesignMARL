import random

import torch
# import numpy as np
import matplotlib.pyplot as plt


# all j are independent

# no collision
# j' reach the goal of j, will not receive any reward, and the goal will not disappear

class reaching_goals_env(object):
    def __init__(self, config, device):
        self.map_height, self.map_width, self.max_step, self.dim_action = config.map_height, config.map_width, config.max_step, config.dim_action
        self.reward_amplifier, self.punish_amplifier = config.reward_amplifier, config.punish_amplifier
        if hasattr(config, 'nj'):
            self.nj = config.nj
        else:
            self.nj = 1

        self.device = device

        self.color_map = {
            'agent': [20, 20, 220],  # blue
            'sender_apple': [220, 20, 20],  # red
            'receiver_apple': [20, 220, 20],  # green
            'message': [255, 255, 255],  # white
        }

        # up, down, left, right
        self.action_to_delta_pos_map = torch.tensor([[-1, 0],
                                                     [+1, 0],
                                                     [0, -1],
                                                     [0, +1]], dtype=torch.int32, device=self.device)
        self.reset()

        return

    def reset(self):
        self.reset_agents()  # agent -> receiver

        self.receiver_apple_position, self.receiver_apple_channel = self.generate_apple()
        while self.check_reached('receiver'):
            self.receiver_apple_position, self.receiver_apple_channel = self.generate_apple()

        self.sender_apple_position, self.sender_apple_channel = self.generate_apple()
        while self.check_reached('sender'):
            self.sender_apple_position, self.sender_apple_channel = self.generate_apple()
        self.step_i = 0

        state = torch.cat([self.agent_channel.unsqueeze(dim=0),
                           self.sender_apple_channel.unsqueeze(dim=0),
                           self.receiver_apple_channel.unsqueeze(dim=0), ])
        return state.unsqueeze(dim=0)

    def generate_objects(self, n_obj):
        height, width = self.map_height, self.map_width

        positions_int = torch.randint(height * width, size=(n_obj,), device=self.device)
        channels = torch.nn.functional.one_hot(positions_int).view(n_obj, height, width)
        # channel_np = np.array(channels)

        x = torch.floor(positions_int / width).long()
        y = positions_int % width
        positions = torch.cat([x.unsqueeze(dim=1), y.unsqueeze(dim=1)], dim=1)

        return positions, channels

    def generate_map(self):
        self.map_color = self.generate_color_channels(self.agent_channel, self.color_map['agent']) \
                         + self.generate_color_channels(self.sender_apple_channel, self.color_map['sender_apple']) \
                         + self.generate_color_channels(self.receiver_apple_channel, self.color_map['receiver_apple'])

        # self.map_color_np = np.array(self.map_color)

    def generate_color_channels(self, channel, color_map):
        color_map = torch.tensor(color_map, device=self.device)

        return channel.unsqueeze(dim=0).repeat(3, 1, 1) \
               * color_map.unsqueeze(dim=-1).repeat(1, self.map_height).unsqueeze(dim=-1).repeat(1, 1, self.map_width)

    def render(self, step, type='before', message=None, phi=None, pi=None, filename=None, ):
        assert type in ['before', 'after']

        plt.clf()
        self.generate_map()

        if phi is None:
            plt.title('t={}\n{} action'.format(step, type))
            plt.imshow(self.map_color.transpose(0, 1).transpose(1, 2).int(), interpolation='nearest')
            if not message is None:
                message_pos = torch.nonzero(message == 1).squeeze(dim=0)[2:]
                plt.plot(int(message_pos[1].int()), int(message_pos[0].int()), marker='o', markersize=20, color='pink')
            if filename is None:
                plt.draw()
                plt.pause(0.1)
            else:
                plt.savefig(filename, dpi=300)
        else:
            assert not message is None
            assert not filename is None
            assert not pi is None

            fig = plt.figure(dpi=300)
            pi_fig = fig.add_subplot(1, 3, 1)
            env_fig = fig.add_subplot(1, 3, 2)
            phi_fig = fig.add_subplot(1, 3, 3)

            pi_fig.set_title('Receiver\'s Policy\n(pi)')
            env_fig.set_title('t={}\n{} action'.format(step, type))
            phi_fig.set_title('Signaling Scheme\n(phi)')

            pi_fig.imshow(pi.detach() * 255, interpolation='nearest', cmap='gray')
            env_fig.imshow(self.map_color.transpose(0, 1).transpose(1, 2).int(), interpolation='nearest')
            phi_fig.imshow(phi.detach().view(self.map_height, self.map_width) * 255, interpolation='nearest',
                           cmap='gray')

            pi_fig.set_xticks(range(self.dim_action))
            pi_fig.set_xticklabels(['up', 'down', 'left', 'right'])
            message_pos = torch.nonzero(message == 1).squeeze(dim=0)[2:]
            env_fig.plot(int(message_pos[1].int()), int(message_pos[0].int()), marker='o', markersize=20, color='pink')

            fig.savefig(filename)

        return

    def check_reached(self, type):
        assert type in ['sender', 'receiver']
        if type == 'sender':
            flag = self.agent_position[0] == self.sender_apple_position[0] and self.agent_position[1] == \
                   self.sender_apple_position[1]
        else:
            flag = self.agent_position[0] == self.receiver_apple_position[0] and self.agent_position[1] == \
                   self.receiver_apple_position[1]
        return flag

    def calculate_distance(self, pos1, pos2):
        distance = ((pos1[0] - pos2[0]) / (self.map_height - 1)) ** 2 + \
                   ((pos1[1] - pos2[1]) / (self.map_width - 1)) ** 2
        return distance ** 0.5

    def step(self, action):
        delta_pos = self.action_to_delta_pos_map[action]
        self.agent_position[0] = self.agent_position[0] + delta_pos[0]
        self.agent_position[1] = self.agent_position[1] + delta_pos[1]

        if self.agent_position[0] < 0 or self.agent_position[0] >= self.map_width:
            self.agent_position[0] = self.agent_position[0] - delta_pos[0]
        if self.agent_position[1] < 0 or self.agent_position[1] >= self.map_height:
            self.agent_position[1] = self.agent_position[1] - delta_pos[1]

        self.agent_channel = torch.zeros(self.map_height, self.map_width, dtype=torch.double, device=self.device)
        self.agent_channel[self.agent_position[0], self.agent_position[1]] = 1
        # self.agent_channel_np = np.array(self.agent_channel)

        if self.check_reached('receiver'):
            receiver_reward = 1 * self.reward_amplifier
            while self.check_reached('receiver'):
                self.receiver_apple_position, self.receiver_apple_channel = self.generate_apple()
        else:
            receiver_reward = - self.calculate_distance(self.agent_position,
                                                        self.receiver_apple_position) / (
                                      2 ** 0.5) * self.punish_amplifier

        if self.check_reached('sender'):
            sender_reward = 1 * self.reward_amplifier
            self.sender_apple_position, self.sender_apple_channel = self.receiver_apple_position, self.receiver_apple_channel
        else:
            sender_reward = - self.calculate_distance(self.agent_position,
                                                      self.sender_apple_position) / (
                                    2 ** 0.5) * self.punish_amplifier
        self.step_i += 1
        done = True if self.step_i >= self.max_step else False

        state = torch.cat([self.agent_channel.unsqueeze(dim=0),
                           self.sender_apple_channel.unsqueeze(dim=0),
                           self.receiver_apple_channel.unsqueeze(dim=0), ])

        return state.unsqueeze(dim=0), [float(sender_reward),
                                        float(receiver_reward)], done


def human_play(config):
    import sys

    step = 0
    env = reaching_goals_env(config)
    env.reset()

    a_map = {
        'w': 0,
        's': 1,
        'a': 2,
        'd': 3
    }

    done = False
    while not done:
        env.render(step, type='before')
        a = sys.stdin.readline().rstrip()

        state, [rs, rr], done = env.step(a_map[a])
        print(rs, rr, done)
        env.render(step, type='after')

    return


def random_move(config):
    env = reaching_goals_env(config)
    env.reset()
    env.render()
    done = False
    total_reward_s = 0
    total_reward_r = 0
    while not done:
        a = random.randint(0, 3)
        state, [rs, rr], done = env.step(a)
        total_reward_s += rs
        total_reward_r += rr
        # print(rs, rr, done)
        env.render()
    print(total_reward_s, total_reward_r)


if __name__ == '__main__':
    from utils.configdict import ConfigDict

    config = ConfigDict()
    config.map_height = 5
    config.map_width = 5
    config.max_step = 50

    config.dim_action = 4
    config.reward_amplifier = 5
    config.punish_amplifier = 1

    human_play(config)

    # for _ in range(1):
    #     random_move()
    print('all done.')
