from utils.configdict import ConfigDict

config_env = ConfigDict()
config_env.name = 'harvest'
config_env.map_name = 'HARVEST_MAP_7'
# config_env.map_name = 'HARVEST_MAP_11'

config_env.max_steps = 50

config_env.obs_height = 5  # maplength-2
# config_env.obs_height = 9 # maplength-2

config_env.obs_width = config_env.obs_height

config_env.global_ref_point = [3,
                               3]  # (maplenth-1)/2 if not None, a fixed global reference frame is used for all agents
# config_env.global_ref_point = [5, 5]  # (maplenth-1)/2 if not None, a fixed global reference frame is used for all agents

config_env.disable_rotation_action = True

config_env.use_agent_position_channel = True
config_env.use_apple_position_channel = True

# =========================

config_env.n_agents = 2
config_env.beam_width = 3
config_env.view_size = int((config_env.obs_height - 1) / 2)
