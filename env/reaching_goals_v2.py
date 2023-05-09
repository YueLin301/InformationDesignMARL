import random

import torch
# import numpy as np
import matplotlib.pyplot as plt


# all j are independent

# no collision
# j' reach the goal of j, will not receive any reward, and the goal will not disappear

class reaching_goals_env(object):
    def __init__(self, config, device_name='cpu'):
        self.map_height, self.map_width, self.max_step, self.dim_action = config.map_height, config.map_width, config.max_step, config.dim_action
        self.reward_amplifier, self.punish_amplifier = config.reward_amplifier, config.punish_amplifier
        if hasattr(config, 'nj'):
            self.nj = config.nj
        else:
            self.nj = 1

        self.device = torch.device(device_name)

        self.n_obj = 2 * self.nj + 1  # n receivers pos, n+1 apples

        self.apple_i_idx = [0]
        self.apple_j_idx = [k + 1 for k in range(self.nj)]
        self.pos_j_idx = [k + 1 + self.nj for k in range(self.nj)]

        self.color_map = torch.tensor([[220, 20, 20],  # apple i: red
                                       [20, 220, 20],  # apple j: green
                                       [20, 20, 220]],  # pos j: blue
                                      device=self.device)
        self.color_map_repeat = self.color_map.unsqueeze(dim=-1).repeat(1, 1, self.map_height) \
            .unsqueeze(dim=-1).repeat(1, 1, 1, self.map_width)

        # up, down, left, right
        self.action_to_delta_pos_map = torch.tensor([[-1, 0],
                                                     [+1, 0],
                                                     [0, -1],
                                                     [0, +1]], dtype=torch.int32, device=self.device)
        self.reset()

        return

    def reset(self):
        self.step_i = 0

        flag_eachj, flag_ij = False, False
        while not flag_ij or not flag_eachj:
            self.positions, self.channels, self.positions_int = self.generate_objects(self.n_obj)

            self.apple_i_position, self.apple_i_channels, self.apple_i_positions_int \
                = self.positions[self.apple_i_idx], self.channels[self.apple_i_idx], self.positions_int[
                self.apple_i_idx]
            self.apple_j_positions, self.apple_j_channels, self.apple_j_positions_int \
                = self.positions[self.apple_j_idx], self.channels[self.apple_j_idx], self.positions_int[
                self.apple_j_idx]
            self.pos_j_positions, self.pos_j_channels, self.pos_j_positions_int \
                = self.positions[self.pos_j_idx], self.channels[self.pos_j_idx], self.positions_int[self.pos_j_idx]

            flag_eachj = int(torch.prod(self.apple_j_positions_int - self.pos_j_positions_int))
            flag_ij = int(torch.prod(self.apple_i_positions_int.repeat(self.nj) - self.pos_j_positions_int))

        state = self.channels.unsqueeze(dim=0)
        return state.detach()

    def generate_objects(self, n_obj):
        height, width = self.map_height, self.map_width

        x = torch.randint(height, size=(n_obj,), device=self.device)
        y = torch.randint(width, size=(n_obj,), device=self.device)
        positions = torch.cat([x.unsqueeze(dim=1), y.unsqueeze(dim=1)], dim=1)

        positions_int = x * width + y
        channels = torch.nn.functional.one_hot(positions_int, num_classes=height * width).view(n_obj, height, width)
        # channel_np = np.array(channels)

        return positions, channels, positions_int

    def generate_map(self):
        apple_j_all_channels = torch.sum(self.apple_j_channels, dim=0).to(bool).to(int).unsqueeze(dim=0)
        pos_j_all_channels = torch.sum(self.pos_j_channels, dim=0).to(bool).to(int).unsqueeze(dim=0)
        all_channels = torch.cat([self.apple_i_channels, apple_j_all_channels, pos_j_all_channels], dim=0)
        self.map_color = torch.clamp_(torch.sum(all_channels * self.color_map_repeat, dim=0), 0, 255)

        # self.map_color_np = np.array(self.map_color)

    def render(self, step=None, type='before', filename=None):
        assert type in ['before', 'after']

        plt.clf()
        self.generate_map()

        plt.title('t={}\n{} action'.format(step, type))
        plt.imshow(self.map_color.transpose(0, 1).transpose(1, 2).int(), interpolation='nearest')

        for i in range(self.nj):
            xi, yi = self.pos_j_positions[i, 0], self.pos_j_positions[i, 1]
            plt.text(yi, xi, str(i), color='grey', size='15')

            xi, yi = self.apple_j_positions[i, 0], self.apple_j_positions[i, 1]
            plt.text(yi, xi, str(i), color='grey', size='15')

        if filename is None:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.savefig(filename, dpi=300)
        return

    def calculate_distance(self, pos1, pos2):
        x1, x2 = pos1[:, 0] / (self.map_height - 1), pos2[:, 0] / (self.map_height - 1)
        y1, y2 = pos1[:, 1] / (self.map_width - 1), pos2[:, 1] / (self.map_width - 1)
        pos1_, pos2_ = torch.cat([x1.unsqueeze(dim=-1), y1.unsqueeze(dim=-1)], dim=-1), \
                       torch.cat([x2.unsqueeze(dim=-1), y2.unsqueeze(dim=-1)], dim=-1)

        distance_all = torch.cdist(pos1_, pos2_, p=2)
        eye = torch.eye(self.nj, device=self.device)

        return torch.sum(distance_all * eye, dim=0)

    def step(self, actions):
        assert hasattr(actions, '__iter__')
        delta_pos = self.action_to_delta_pos_map[actions]
        self.pos_j_positions = self.pos_j_positions + delta_pos

        pos_j_xs = torch.clamp_(self.pos_j_positions[:, 0], 0, self.map_height - 1)
        pos_j_ys = torch.clamp_(self.pos_j_positions[:, 1], 0, self.map_width - 1)

        self.pos_j_positions_int = pos_j_xs * self.map_width + pos_j_ys
        self.pos_j_channels = torch.nn.functional.one_hot(self.pos_j_positions_int,
                                                          num_classes=self.map_height * self.map_width) \
            .view(self.nj, self.map_height, self.map_width)

        # flag_eachj = int(torch.prod(self.apple_j_positions_int - self.pos_j_positions_int))
        # flag_ij = int(torch.prod(self.apple_i_positions_int.repeat(self.nj) - self.pos_j_positions_int))

        flag_all_j_int = self.apple_j_positions_int - self.pos_j_positions_int
        flag_all_j = flag_all_j_int.to(bool).to(int)  # 0 if reached
        flag_ij = torch.prod(self.apple_i_positions_int.repeat(self.nj) - self.pos_j_positions_int).to(bool).to(int)

        distance_jj = self.calculate_distance(self.pos_j_positions, self.apple_j_positions)
        j_rewards = (1 - flag_all_j) * self.reward_amplifier - flag_all_j * self.punish_amplifier * distance_jj

        distance_ij = torch.mean(
            self.calculate_distance(self.pos_j_positions, self.apple_i_position.repeat(self.nj, 1)))
        i_reward = (1 - flag_ij) * self.reward_amplifier - flag_ij * self.punish_amplifier * distance_ij

        flag_eachj = False
        while not flag_eachj:
            apple_j_positions, _, _ = self.generate_objects(self.nj)
            mask_pos = flag_all_j.unsqueeze(dim=-1).repeat(1, 2)
            self.apple_j_positions = apple_j_positions * (1 - mask_pos) + self.apple_j_positions * mask_pos

            x, y = self.apple_j_positions[:, 0], self.apple_j_positions[:, 1]
            self.apple_j_positions_int = x * self.map_width + y
            self.apple_j_channels = torch.nn.functional.one_hot(self.apple_j_positions_int,
                                                                num_classes=self.map_height * self.map_width) \
                .view(self.nj, self.map_height, self.map_width)
            flag_eachj = int(torch.prod(self.apple_j_positions_int - self.pos_j_positions_int))

        while not flag_ij:
            self.apple_i_position, self.apple_i_channels, self.apple_i_positions_int = self.generate_objects(1)
            flag_ij = int(torch.prod(self.apple_i_positions_int.repeat(self.nj) - self.pos_j_positions_int))

        self.step_i += 1
        done = True if self.step_i >= self.max_step else False
        state = self.channels.unsqueeze(dim=0)

        return state.detach(), i_reward, j_rewards.detach(), done


def human_play(config):
    import time

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
        # a = sys.stdin.readline().rstrip()
        actions_str = input("input actions:").split(' ')

        assert len(actions_str) == config.nj
        actions = [a_map[i] for i in actions_str]

        state, rs, rr, done = env.step(actions)
        print(rs, rr, done)
        env.render(step, type='after')
        step += 1
        time.sleep(1)

    return


if __name__ == '__main__':
    from utils.configdict import ConfigDict

    config = ConfigDict()
    config.map_height = 6
    config.map_width = 7
    config.max_step = 50

    config.dim_action = 4
    config.reward_amplifier = 5
    config.punish_amplifier = 1

    config.nj = 4

    human_play(config)

    # for _ in range(1):
    #     random_move()
    print('all done.')
