import torch
import torch.nn as nn
import os

default_dir = './results'


class net_base(nn.Module):
    def __init__(self, n_channels, config, name=None, chkpt_dir=default_dir, device=None):
        super().__init__()
        self.n_channels = n_channels
        # padding for keeping the width and height of input unchanged: kernel=3, padding=1; kernel=5, padding= 2; ...
        self.conv_layer = nn.Sequential(
            nn.Conv2d(n_channels, config.nn.n_filters, config.nn.kernel, config.nn.stride,
                      padding=int((config.nn.kernel - 1) / 2), dtype=torch.double), nn.ReLU(),
        )

        obs_vector = config.env.obs_height * config.env.obs_width * config.nn.n_filters
        self.mlp = nn.Sequential(
            nn.Linear(obs_vector, config.nn.hidden_width, dtype=torch.double), nn.ReLU(),
            nn.Linear(config.nn.hidden_width, config.nn.hidden_width, dtype=torch.double), nn.ReLU(),
        )

        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, config.main.exp_name, name)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.mlp(self.conv_layer(x).view(x.shape[0], -1))

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        return

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class actor(net_base):
    def __init__(self, n_channels, config, name='actor', chkpt_dir=default_dir, device=None):
        super(actor, self).__init__(n_channels, config, name=name, chkpt_dir=chkpt_dir, device=device)

        self.action_dim = config.env.dim_action
        self.output_layer = nn.Sequential(nn.Linear(config.nn.hidden_width, config.nn.hidden_width, dtype=torch.double),
                                          nn.ReLU(),
                                          nn.Linear(config.nn.hidden_width, self.action_dim, dtype=torch.double), )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        y1 = super(actor, self).forward(x)
        y2 = self.output_layer(y1)
        return y2

    def get_action_prob_logprob_stochastic(self, input):
        a_prob = self.softmax(self.forward(input))

        a_int = torch.multinomial(a_prob, num_samples=1, replacement=True, )[:, 0]
        distribution = torch.distributions.Categorical(a_prob)
        return a_int, a_prob, distribution.log_prob(a_int)


class critic(net_base):
    def __init__(self, n_channels, config, critic_type, name='critic', chkpt_dir=default_dir, device=None):
        super(critic, self).__init__(n_channels, config, name=name, chkpt_dir=chkpt_dir, device=device)

        assert critic_type in ['G', 'Q']
        self.critic_type = critic_type
        self.action_dim = config.env.dim_action
        self.output_dim = self.action_dim ** config.env.n_agents if critic_type == 'G' else self.action_dim
        self.output_layer = nn.Sequential(nn.Linear(config.nn.hidden_width, config.nn.hidden_width, dtype=torch.double),
                                          nn.ReLU(),
                                          nn.Linear(config.nn.hidden_width, self.output_dim, dtype=torch.double))

    def forward(self, obs, *a):
        y1 = super(critic, self).forward(obs)
        QorG_tables = self.output_layer(y1)

        idx = a[0] * self.action_dim + a[1] if self.critic_type == 'G' else a[0]
        QorG = QorG_tables[range(len(idx)), idx]

        return QorG


# The signaling net should be stochastic.
# Its output is a continuous stochastic distribution, or PDF. The dim is inf.
# For now, the Gaussian distribution is used for testing. The outputs are the \mu and \sigma, for every pixel.
class signaling_net(net_base):
    def __init__(self, config, name='signaling_net', chkpt_dir=default_dir, device=None):
        super(signaling_net, self).__init__(config.sender.n_channels, config, name=name, chkpt_dir=chkpt_dir,
                                            device=device)

        self.message_height = config.env.obs_height
        self.message_width = config.env.obs_width

        # logits of 0 or 1, for every pixel
        self.output_dim = self.message_height * self.message_width * 2

        self.output_layer_logits = nn.Sequential(
            nn.Linear(config.nn.hidden_width, config.nn.hidden_width, dtype=torch.double), nn.ReLU(),
            nn.Linear(config.nn.hidden_width, self.output_dim, dtype=torch.double),
        )

    def forward(self, x):
        y = super(signaling_net, self).forward(x)
        logits = self.output_layer_logits(y).view(self.output_dim // 2, 2)
        phi = torch.softmax(logits, dim=-1)
        return phi
