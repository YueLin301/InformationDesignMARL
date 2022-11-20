"""Wrapper around Sequential Social Dilemma environment."""

from env import maps
from social_dilemmas.constants import HARVEST_MAP
from env.harvest_base import HarvestEnv
import numpy as np


class Env(object):

    def __init__(self, config_env):

        self.config = config_env
        self.dim_obs = [self.config.obs_height,
                        self.config.obs_width, 3]
        self.max_steps = self.config.max_steps

        # Original space (not necessarily in this order, see
        # the original ssd files):
        # no-op, up, down, left, right, turn-ccw, turn-cw, penalty,
        if self.config.disable_rotation_action:
            self.l_action = 5
            # left, right, up, down, no-op,
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, }

        config_env.dim_action = self.l_action
        self.n_agents = self.config.n_agents

        if self.config.map_name == 'HARVEST_MAP_11':
            ascii_map = maps.HARVEST_MAP_11
        elif self.config.map_name == 'HARVEST_MAP_7':
            ascii_map = maps.HARVEST_MAP_7

        self.env = HarvestEnv(ascii_map=ascii_map,
                              num_agents=self.n_agents, render=False,
                              global_ref_point=self.config.global_ref_point,
                              view_size=self.config.view_size,
                              beam_width=self.config.beam_width)

        self.steps = 0

    def process_obs(self, obs_dict):
        obs_allagent = [obs / 256.0 for obs in list(obs_dict.values())]

        obs_allagent = self.generate_obs_with_channels(self.config.use_agent_position_channel, obs_allagent,
                                                       self.env.agent_pos)
        obs_allagent = self.generate_obs_with_channels(self.config.use_apple_position_channel, obs_allagent,
                                                       self.get_item_pos('apple'))
        return obs_allagent

    def reset(self):
        """Resets the environemnt.

        Returns:
            List of agent observations
        """
        obs = self.env.reset()
        self.steps = 0

        return self.process_obs(obs)

    def step(self, actions):
        """Takes a step in env.

        Args:
            actions: list of integers

        Returns:
            List of observations, list of rewards, done, info
        """
        actions = [self.map_to_orig[a] for a in actions]
        actions_dict = {'agent-%d' % idx: actions[idx]
                        for idx in range(self.n_agents)}

        # all objects returned by env.step are dicts
        obs_next, rewards, dones, info = self.env.step(actions_dict)
        self.steps += 1

        obs_next = self.process_obs(obs_next)
        rewards = list(rewards.values())

        # done = dones['__all__']  # apparently they hardcode done to False
        done = dones['__all__'] or self.steps == self.max_steps

        return obs_next, rewards, done, info

    def render(self):
        self.env.render()

    def generate_obs_with_channels(self, pls_generate, obs_allagent, position_allitem):
        obs_with_position_allagent = obs_allagent
        if pls_generate:
            for agent_i in range(len(obs_allagent)):
                agent_name = 'agent-' + str(agent_i)
                orientation_i = self.env.agents[agent_name].get_orientation()
                obs = obs_allagent[agent_i]
                position_channel = np.zeros_like(obs[:, :, 0])
                for position_i in position_allitem:
                    row = position_i[0] - int((len(self.env.test_map) - self.config.obs_height) / 2)
                    col = position_i[1] - int((len(self.env.test_map[0]) - self.config.obs_width) / 2)
                    position_channel[row][col] = 1
                position_channel = self.env.rotate_view(orientation_i, position_channel)
                position_channel = np.expand_dims(position_channel, axis=2)
                obs_with_position_allagent[agent_i] = np.concatenate([obs, position_channel], axis=2)

        return obs_with_position_allagent

    # 里面自带的那个apple_points和waste_points是base_map里的，不是实时的，真要找还得看test_map来找
    def get_item_pos(self, item):
        pos = []
        for row_i in range(len(self.env.test_map)):
            for col_j in range(len(self.env.test_map[row_i])):
                if item == 'apple':
                    if self.env.test_map[row_i][col_j] == 'A':
                        pos.append((row_i, col_j))
        return pos


if __name__ == '__main__':
    from configs import configfile_harvest_test

    config = configfile_harvest_test.get_config(wandb_config=False)

    env = Env(config.env)

    obs_list = env.reset()
    env.render()

    actions_list = [1, 2]
    obs_next_list, reward_list, done, info = env.step(actions_list)


    print('haha')
