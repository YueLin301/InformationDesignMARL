from DIAL_origin.agent import DRU
from exp_reaching_goals.reaching_goals_utils import init_wandb
from exp_reaching_goals.network import critic


class sender_DIAL():
    def __init__(self, config, device):
        self.name = 'sender'
        self.config = config
        self.device = device

        self.critic = critic
