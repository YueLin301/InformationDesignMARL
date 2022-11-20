"""Wrapper around Sequential Social Dilemma environment."""

from env import maps
# from social_dilemmas.constants import CLEANUP_MAP
from social_dilemmas.envs.cleanup import CleanupEnv
import numpy as np


class Env(object):

    def __init__(self, config_env):

        self.config = config_env
        self.dim_obs = [self.config.obs_height,
                        self.config.obs_width, 3]
        self.max_steps = self.config.max_steps

        self.cleaning_penalty = self.config.cleaning_penalty
        # Original space (not necessarily in this order, see
        # the original ssd files):
        # no-op, up, down, left, right, turn-ccw, turn-cw, penalty, clean
        if (self.config.disable_left_right_action and
                self.config.disable_rotation_action):
            self.l_action = 4
            self.cleaning_action_idx = 3
            # up, down, no-op, clean
            self.map_to_orig = {0: 2, 1: 3, 2: 4, 3: 8}
        elif self.config.disable_left_right_action:
            self.l_action = 6
            self.cleaning_action_idx = 5
            # up, down, no-op, rotate cw, rotate ccw, clean
            self.map_to_orig = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8}
        elif self.config.disable_rotation_action:
            self.l_action = 6
            self.cleaning_action_idx = 5
            # left, right, up, down, no-op, clean
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 8}
        else:  # full action space except penalty beam
            self.l_action = 8
            self.cleaning_action_idx = 7
            # Don't allow penalty beam
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8}
        config_env.dim_action = self.l_action

        self.n_agents = self.config.n_agents

        if self.config.map_name == 'cleanup_small_sym':
            ascii_map = maps.CLEANUP_SMALL_SYM
        elif self.config.map_name == 'cleanup_10x10_sym':
            ascii_map = maps.CLEANUP_10x10_SYM
        elif self.config.map_name == 'CLEANUP_SMALL_new':
            ascii_map = maps.CLEANUP_SMALL_new
        elif self.config.map_name == 'CLEANUP_easy':
            ascii_map = maps.CLEANUP_easy
        elif self.config.map_name == 'CLEANUP_big':
            ascii_map = maps.CLEANUP_big
        elif self.config.map_name == 'CLEANUP_SMALL_rightup':
            ascii_map = maps.CLEANUP_SMALL_rightup
        elif self.config.map_name == 'CLEANUP_11x11_allapple':
            ascii_map = maps.CLEANUP_11x11_allapple
        elif self.config.map_name == 'CLEANUP_11x11_rightwaste':
            ascii_map = maps.CLEANUP_11x11_rightwaste
        elif self.config.map_name == 'CLEANUP_11x11_origin':
            ascii_map = maps.CLEANUP_11x11_origin
        elif self.config.map_name == 'CLEANUP_7x7_allapple':
            ascii_map = maps.CLEANUP_7x7_allapple
        elif self.config.map_name == 'CLEANUP_MAP':
            ascii_map = maps.CLEANUP_MAP  # 25*18
        elif self.config.map_name == 'CLEANUP_SMALL_same_col':
            ascii_map = maps.CLEANUP_SMALL_same_col


        self.env = CleanupEnv(ascii_map=ascii_map,
                              num_agents=self.n_agents, render=False,
                              shuffle_spawn=self.config.shuffle_spawn,
                              global_ref_point=self.config.global_ref_point,
                              view_size=self.config.view_size,
                              random_orientation=self.config.random_orientation,
                              cleanup_params=self.config.cleanup_params,
                              beam_width=self.config.beam_width)

        self.steps = 0

    def process_obs(self, obs_dict):
        obs_allagent = [obs / 256.0 for obs in list(obs_dict.values())]

        obs_allagent = self.generate_obs_with_channels(self.config.use_agent_position_channel, obs_allagent,
                                                       self.env.agent_pos)
        obs_allagent = self.generate_obs_with_channels(self.config.use_apple_position_channel, obs_allagent,
                                                       self.get_item_pos('apple'))
        obs_allagent = self.generate_obs_with_channels(self.config.use_waste_position_channel, obs_allagent,
                                                       self.get_item_pos('waste'))
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
        if self.cleaning_penalty > 0:
            for idx in range(self.n_agents):
                if actions[idx] == 8:
                    rewards[idx] -= self.cleaning_penalty

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
                elif item == 'waste':
                    if self.env.test_map[row_i][col_j] == 'H':
                        pos.append((row_i, col_j))
        return pos
