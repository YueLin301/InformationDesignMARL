import random

import torch
import numpy as np
import matplotlib.pyplot as plt


class reaching_goals_env(object):
    def __init__(self, config):
        assert config.map_height % 2 and config.map_width % 2
        self.map_height, self.map_width, self.max_step, self.aligned_object, self.bounded = config.map_height, config.map_width, config.max_step, config.aligned_object, config.bounded

        self.color_map = {
            'agent': [20, 20, 220],  # blue
            'sender_apple': [220, 20, 20],  # red
            'receiver_apple': [20, 220, 20],  # green
            'message': [255, 255, 255],  # white
        }

        # up, down, left, right
        self.action_to_delta_pos_map = {
            0: [-1, 0],
            1: [+1, 0],
            2: [0, -1],
            3: [0, +1],
        }

        self.reset()

        return

    def reset(self):
        self.reset_agent()  # agent -> receiver
        self.receiver_apple_position, self.receiver_apple_channel, self.receiver_apple_channel_np = self.generate_apple()
        if not self.aligned_object:
            self.sender_apple_position, self.sender_apple_channel, self.sender_apple_channel_np = self.generate_apple()
        else:
            self.sender_apple_position, self.sender_apple_channel, self.sender_apple_channel_np = self.receiver_apple_position, self.receiver_apple_channel, self.receiver_apple_channel_np
        self.step_i = 0

        if self.aligned_object:
            state = torch.cat([self.agent_channel.unsqueeze(dim=0),
                               self.receiver_apple_channel.unsqueeze(dim=0), ])
        else:
            state = torch.cat([self.agent_channel.unsqueeze(dim=0),
                               self.sender_apple_channel.unsqueeze(dim=0),
                               self.receiver_apple_channel.unsqueeze(dim=0), ])
        return state.unsqueeze(dim=0)

    def generate_map(self):
        self.map_color = self.generate_color_channels(self.agent_channel, self.color_map['agent']) \
                         + self.generate_color_channels(self.sender_apple_channel, self.color_map['sender_apple']) \
                         + self.generate_color_channels(self.receiver_apple_channel, self.color_map['receiver_apple'])

        self.map_color_np = np.array(self.map_color)

    def generate_color_channels(self, channel, color_map):
        color_map = torch.tensor(color_map)

        return channel.unsqueeze(dim=0).repeat(3, 1, 1) \
               * color_map.unsqueeze(dim=-1).repeat(1, self.map_height).unsqueeze(dim=-1).repeat(1, 1, self.map_width)

    def reset_agent(self):
        self.agent_position = [(self.map_height - 1) // 2, (self.map_width - 1) // 2]
        self.agent_channel = torch.zeros(self.map_height, self.map_width, dtype=torch.double)
        self.agent_channel[self.agent_position[0], self.agent_position[1]] = 1
        self.agent_channel_np = np.array(self.agent_channel)

    def generate_apple(self):
        apple_position_flatten = torch.randint(self.map_height * self.map_width, size=(1,))
        apple_position = [torch.floor(apple_position_flatten / self.map_width).long(),
                          apple_position_flatten % self.map_width]

        apple_channel = torch.zeros(self.map_height, self.map_width, dtype=torch.double)
        apple_channel[apple_position[0], apple_position[1]] = 1
        apple_channel_np = np.array(apple_channel)

        return apple_position, apple_channel, apple_channel_np

    def render(self, message_pos=None, filename=None, ):
        plt.clf()
        self.generate_map()
        plt.imshow(self.map_color.transpose(0, 1).transpose(1, 2).int(), interpolation='nearest')
        if not message_pos is None:
            plt.plot(int(message_pos[0].int()), int(message_pos[1].int()), marker='o', markersize=20, color='pink')
        if filename is None:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.savefig(filename)
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

    def step(self, action):
        delta_pos = self.action_to_delta_pos_map[action]
        self.agent_position[0] = self.agent_position[0] + delta_pos[0]
        self.agent_position[1] = self.agent_position[1] + delta_pos[1]

        if not self.bounded:
            if self.agent_position[0] < 0:
                self.agent_position[0] = self.agent_position[0] + self.map_width
            if self.agent_position[0] >= self.map_width:
                self.agent_position[0] = self.agent_position[0] - self.map_width

            if self.agent_position[1] < 0:
                self.agent_position[1] = self.agent_position[1] + self.map_height
            if self.agent_position[1] >= self.map_height:
                self.agent_position[1] = self.agent_position[1] - self.map_height
        else:
            if self.agent_position[0] < 0 or self.agent_position[0] >= self.map_width:
                self.agent_position[0] = self.agent_position[0] - delta_pos[0]
            if self.agent_position[1] < 0 or self.agent_position[1] >= self.map_height:
                self.agent_position[1] = self.agent_position[1] - delta_pos[1]

        self.agent_channel = torch.zeros(self.map_height, self.map_width, dtype=torch.double)
        self.agent_channel[self.agent_position[0], self.agent_position[1]] = 1
        self.agent_channel_np = np.array(self.agent_channel)

        if self.check_reached('receiver'):
            receiver_reward = 1
            while self.check_reached('receiver'):
                self.receiver_apple_position, self.receiver_apple_channel, self.receiver_apple_channel_np = self.generate_apple()
        else:
            receiver_reward = 0

        if self.check_reached('sender'):
            sender_reward = 1
            if not self.aligned_object:
                while self.check_reached('sender'):
                    self.sender_apple_position, self.sender_apple_channel, self.sender_apple_channel_np = self.generate_apple()
            else:
                self.sender_apple_position, self.sender_apple_channel, self.sender_apple_channel_np = self.receiver_apple_position, self.receiver_apple_channel, self.receiver_apple_channel_np
        else:
            sender_reward = 0

        self.step_i += 1
        if self.step_i >= self.max_step:
            done = True
        else:
            done = False

        if self.aligned_object:
            state = torch.cat([self.agent_channel.unsqueeze(dim=0),
                               self.receiver_apple_channel.unsqueeze(dim=0), ])
        else:
            state = torch.cat([self.agent_channel.unsqueeze(dim=0),
                               self.sender_apple_channel.unsqueeze(dim=0),
                               self.receiver_apple_channel.unsqueeze(dim=0), ])
        return state.unsqueeze(dim=0), [sender_reward, receiver_reward], done


def human_play(config):
    import sys

    env = reaching_goals_env(config)
    env.reset()
    env.render()

    a_map = {
        'w': 0,
        's': 1,
        'a': 2,
        'd': 3
    }

    done = False
    while not done:
        a = sys.stdin.readline().rstrip()

        state, [rs, rr], done = env.step(a_map[a])
        print(rs, rr, done)
        env.render()

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
    config.map_height = 7
    config.map_width = 7
    config.max_step = 50
    config.aligned_object = True
    config.bounded = True

    human_play(config)

    # for _ in range(1):
    #     random_move()
    print('all done.')
