import torch
import torch.nn as nn
import os


class net_base(nn.Module):
    def __init__(self, n_channels, config, belongto, name, device=None):
        super().__init__()
        self.n_channels = n_channels
        # padding for keeping the width and height of input unchanged: kernel=3, padding=1; kernel=5, padding= 2; ...
        self.conv_layer = nn.Sequential(
            nn.Conv2d(n_channels, config.nn.n_filters, config.nn.kernel, config.nn.stride,
                      padding=int((config.nn.kernel - 1) / 2), dtype=torch.double), nn.ReLU(),
        )

        obs_vector = config.env.map_height * config.env.map_width * config.nn.n_filters
        self.mlp = nn.Sequential(
            nn.Linear(obs_vector, config.nn.hidden_width, dtype=torch.double), nn.ReLU(),
            nn.Linear(config.nn.hidden_width, config.nn.hidden_width, dtype=torch.double), nn.ReLU(),
        )

        # assert belongto in ['sender', 'receiver']
        self.belongto = belongto
        self.name = name
        self.checkpoint_file = os.path.join(config.path.saved_models, config.main.exp_name, belongto, name)
        # print(os.getcwd())
        if not os.path.exists(os.path.join(config.path.saved_models, config.main.exp_name, belongto)):
            os.makedirs(os.path.join(config.path.saved_models, config.main.exp_name, belongto), exist_ok=True)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.mlp(self.conv_layer(x).view(x.shape[0], -1))

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        return

    def load_checkpoint(self, path=None):
        if not path:
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        else:
            load_path = os.path.join(path, self.belongto, self.name)
            self.load_state_dict(torch.load(load_path, map_location=self.device))


class actor(net_base):
    def __init__(self, n_channels, config, belongto, name='actor', device=None):
        super(actor, self).__init__(n_channels, config, belongto, name=name, device=device)

        self.action_dim = config.env.dim_action
        self.output_layer = nn.Sequential(nn.Linear(config.nn.hidden_width, self.action_dim, dtype=torch.double), )
        self.softmax = nn.Softmax(dim=-1)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        y1 = super(actor, self).forward(x)
        y2 = self.output_layer(y1)
        return y2

    def get_a_and_pi(self, input):
        a_prob = self.softmax(self.forward(input))
        a_int = torch.multinomial(a_prob, num_samples=1, replacement=True, )[:, 0]
        return a_int, a_prob


class critic(net_base):
    def __init__(self, input_n_channels, output_dims, config, belongto, name='critic', device=None):
        super(critic, self).__init__(input_n_channels, config, belongto, name=name, device=device)

        # self.action_dim = config.env.dim_action
        self.output_layer = nn.Sequential(nn.Linear(config.nn.hidden_width, output_dims, dtype=torch.double))

        self.device = device
        self.to(self.device)

    def forward(self, obs, *a):
        y1 = super(critic, self).forward(obs)
        QorG_tables = self.output_layer(y1)

        return QorG_tables


class critic_embedding(net_base):
    def __init__(self, input_n_channels, output_dims, config, belongto, name='critic', device=None):
        super(critic_embedding, self).__init__(input_n_channels, config, belongto, name=name, device=device)

        # state: (2 * nj +1) * height * width
        # joint a: 4 * nj
        self.height = config.env.map_height
        self.width = config.env.map_width
        self.a_embedding = torch.nn.Linear(config.env.nj, self.height * self.width,  # output_dim
                                           dtype=torch.double, device=device)

        # self.action_dim = config.env.dim_action
        self.output_layer = nn.Sequential(nn.Linear(config.nn.hidden_width, output_dims, dtype=torch.double))

        self.device = device
        self.to(self.device)

    def forward(self, obs):
        y1 = super(critic_embedding, self).forward(obs)
        QorG_tables = self.output_layer(y1)

        return QorG_tables

    def wrapped_forward(self, obs, aj):
        aj_all = self.a_embedding(aj.to(torch.double)).view(obs.size()[0], 1, self.height, self.width)
        sa_all = torch.cat([obs, aj_all], dim=1)
        G = self.forward(sa_all)

        return G


# The signaling net should be stochastic.
# Its output is a continuous stochastic distribution, or PDF. The dim is inf.
# For now, the Gaussian distribution is used for testing. The outputs are the \mu and \sigma, for every pixel.
class signaling_net(net_base):
    def __init__(self, config, name='signaling_net', device=None):
        self.n_channels = config.n_channels.obs_sender
        super(signaling_net, self).__init__(self.n_channels, config, belongto='sender', name=name,
                                            device=device)

        self.message_height, self.message_width, self.nj = config.env.map_height, config.env.map_width, config.env.nj

        self.output_dim = self.message_height * self.message_width
        self.output_layer_logits = nn.Sequential(
            nn.Linear(config.nn.hidden_width, self.output_dim * self.nj, dtype=torch.double),
        )

        self.device = device
        self.to(self.device)

    def forward(self, x):
        y = super(signaling_net, self).forward(x)
        logits = self.output_layer_logits(y).view(x.shape[0], self.nj, self.output_dim)
        phi = torch.softmax(logits, dim=-1)

        return logits, phi
