import torch
import torch.nn as nn
import os

default_dir = './results'


class net_base(nn.Module):
    def __init__(self, n_channels, config, agent_id=None, name=None, chkpt_dir=default_dir, device=None,
                 howmany_linear=3):
        super().__init__()
        # padding for keeping the width and height of input unchanged: kernel=3, padding=1; kernel=5, padding= 2; ...
        self.conv_layer = nn.Sequential(
            nn.Conv2d(n_channels, config.nn.n_filters, config.nn.kernel, config.nn.stride,
                      padding=int((config.nn.kernel - 1) / 2), dtype=torch.double), nn.ReLU(),
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(config.env.obs_vector, config.nn.n_h2, dtype=torch.double), nn.ReLU(),
            # nn.Linear(config.nn.n_h2, config.nn.n_h2, dtype=torch.double)
        )

        self.rnn_required = config.nn.rnn_required
        if config.nn.rnn_required:
            self.rnn_GRU_layer = nn.GRU(input_size=config.nn.n_h2, hidden_size=config.nn.rnnGRU_hiddennum,
                                        batch_first=True, dtype=torch.double)
            self.rnn_GRU_output_linear = nn.Linear(config.nn.rnnGRU_hiddennum, config.nn.n_h2, dtype=torch.double)

        if howmany_linear == 2:
            self.mlp2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(config.nn.n_h2, config.nn.n_h2, dtype=torch.double), nn.ReLU(),
            )
        elif howmany_linear == 3:
            self.mlp2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(config.nn.n_h2, config.nn.n_h2, dtype=torch.double), nn.ReLU(),
                nn.Linear(config.nn.n_h2, config.nn.n_h2, dtype=torch.double), nn.ReLU(),
            )
        elif howmany_linear == 4:
            self.mlp2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(config.nn.n_h2, config.nn.n_h2, dtype=torch.double), nn.ReLU(),
                nn.Linear(config.nn.n_h2, config.nn.n_h2, dtype=torch.double), nn.ReLU(),
                nn.Linear(config.nn.n_h2, config.nn.n_h2, dtype=torch.double), nn.ReLU(),
            )

        self.name = name
        self.checkpoint_dir = os.path.join(chkpt_dir, config.main.exp_name)
        if agent_id != None:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, 'agent_' + str(agent_id), name)
        else:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.to(self.device)

    def forward(self, x):
        if self.rnn_required:
            y1 = self.conv_layer(x).view(x.shape[0], -1)
            y2 = self.mlp1(y1)
            y3, hidden_state = self.rnn_GRU_layer(y2.unsqueeze(dim=0))  # hidden state不用显式传入，后面也不用，所以不管了
            y4 = self.rnn_GRU_output_linear(y3)
            result = self.mlp2(y4).squeeze(dim=0)
        else:
            result = self.mlp2(self.mlp1(self.conv_layer(x).view(x.shape[0], -1)))
        return result

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        return

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class actor_ind(net_base):
    def __init__(self, n_channels, config, agent_id=None, name='actor_ind', chkpt_dir=default_dir, device=None):
        super(actor_ind, self).__init__(n_channels, config, agent_id=agent_id, name=name, chkpt_dir=chkpt_dir,
                                        device=device, howmany_linear=config.nn.actor_howmany_linear)

        self.action_dim = config.env.dim_action
        self.output_layer = nn.Sequential(nn.Linear(config.nn.n_h2, self.action_dim, dtype=torch.double), )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        y1 = super(actor_ind, self).forward(x)
        y2 = self.output_layer(y1)
        return y2

    def get_action_prob_logprob_stochastic(self, obs_and_message, epsilon=0):
        a_prob = self.softmax(self.forward(obs_and_message))
        a_prob = (1 - epsilon) * a_prob + epsilon / self.action_dim
        # current_generator_state = torch.Generator().get_state()
        a_int = torch.multinomial(a_prob, num_samples=1, replacement=True, )[:, 0]  # 根据概率，有放回地采样
        distribution = torch.distributions.Categorical(a_prob)
        return a_int, a_prob, distribution.log_prob(a_int)

    def get_action_prob_logprob_deterministic(self, obs_and_message, ):
        '''用的action mask来做的，只要给不是最大值的动作加上绝对值很大的负数，然后再传进softmax就可以了，其他一样'''
        logits = self.forward(obs_and_message)
        a_max = torch.max(logits, dim=1)
        mask = torch.ones_like(logits) * (-inf)
        for i in range(len(logits)):
            mask[i, a_max.indices[i]] = 0
        logits = logits + mask

        a_prob = self.softmax(logits)
        a_int = torch.multinomial(a_prob, num_samples=1, replacement=True)[:,
                0]  # sampling according to the distribution
        distribution = torch.distributions.Categorical(a_prob)
        return a_int, a_prob, distribution.log_prob(a_int)


class critic_ind(net_base):

    def __init__(self, n_channels, config, agent_id=None, name='critic_ind', chkpt_dir=default_dir, device=None):
        super(critic_ind, self).__init__(n_channels, config, agent_id=agent_id, name=name, chkpt_dir=chkpt_dir,
                                         device=device, howmany_linear=config.nn.critic_howmany_linear)

        self.output_layer = nn.Sequential(nn.Linear(config.nn.n_h2, 1, dtype=torch.double), )

    def forward(self, x):
        y1 = super(critic_ind, self).forward(x)
        y2 = self.output_layer(y1)
        return y2


class signaling_net_deterministic(net_base):
    def __init__(self, n_channels, config, agent_id=None, name='signaling_net_ind', chkpt_dir=default_dir, device=None):
        super(signaling_net_deterministic, self).__init__(n_channels, config, agent_id=agent_id, name=name,
                                                          chkpt_dir=chkpt_dir,
                                                          device=device,
                                                          howmany_linear=config.nn.signaling_net_howmany_linear)

        self.messages_shape = config.env.messages_shape[1:]

        self.output_layer = nn.Sequential(
            nn.Linear(config.nn.n_h2, (config.env.n_agents - 1) * config.env.obs_height * config.env.obs_width,
                      dtype=torch.double), nn.ReLU(), )

    def forward(self, x):
        y1 = super(signaling_net_deterministic, self).forward(x)
        y2 = self.output_layer(y1)
        self.messages_shape.insert(0, x.shape[0])  # This is the length of the input batch.
        batch_messages_shape = torch.Size(self.messages_shape)
        self.messages_shape.pop(0)
        return y2.view(batch_messages_shape)


'''
# The signaling net should be stochastic.
# Its output is a continuous stochastic distribution, or PDF. The dim is inf.
# For now, the Gaussian distribution is used for testing. The outputs are the \mu and \sigma, for every pixel.
# 尝试1：不行，vmap不支持
class signaling_net(net_base):
    def __init__(self, n_channels, config, agent_id=None, name='signaling_net_ind', chkpt_dir=default_dir, device=None):
        super(signaling_net, self).__init__(n_channels, config, agent_id=agent_id, name=name, chkpt_dir=chkpt_dir,
                                            device=device, howmany_linear=config.nn.signaling_net_howmany_linear)

        self.messages_num = config.env.n_agents - 1
        self.message_height = config.env.obs_height
        self.message_width = config.env.obs_width

        self.messages_shape = config.env.messages_shape[1:]  # 这个是乘了之后的
        self.distribution_description_dim = config.env.message_distribution_description_dim
        self.output_dim = self.messages_num \
                          * self.message_height * self.message_width * self.distribution_description_dim

        self.output_layer = nn.Sequential(
            nn.Linear(config.nn.n_h2, self.output_dim, dtype=torch.double),
            nn.Sigmoid(),  # 均值在0到1，主要是标准差要大于0，所以用了sigmoid
        )

    def forward(self, x):
        y1 = super(signaling_net, self).forward(x)
        y2 = self.output_layer(y1)

        messages_dist_descript_shape = torch.Size([self.distribution_description_dim, x.shape[0]]
                                                  + self.messages_shape.copy())
        messages_distribution_description = y2.view(messages_dist_descript_shape)

        # So far the distribution is Gaussian.
        batch_lmessages_dist = torch.distributions.Normal(messages_distribution_description[0],
                                                          messages_distribution_description[1])

        batch_messages_foreveryoneelse = batch_lmessages_dist.rsample(torch.Size([1]))
        shape = torch.Size([x.shape[0], self.messages_num, self.message_height, self.message_width])
        return batch_messages_foreveryoneelse.squeeze(dim=0).view(shape)

# 尝试2：还是不行，vmap还是不支持
class signaling_net(net_base):
    def __init__(self, n_channels, config, agent_id=None, name='signaling_net_ind', chkpt_dir=default_dir, device=None):
        super(signaling_net, self).__init__(n_channels, config, agent_id=agent_id, name=name, chkpt_dir=chkpt_dir,
                                            device=device, howmany_linear=config.nn.signaling_net_howmany_linear)

        self.messages_num = config.env.n_agents - 1
        self.message_height = config.env.obs_height
        self.message_width = config.env.obs_width

        self.messages_shape = config.env.messages_shape[1:]  # 这个是乘了之后的
        self.distribution_description_dim = config.env.message_distribution_description_dim  # 目前只是2，只用gaussian
        # self.output_dim = self.messages_num \
        #                   * self.message_height * self.message_width * self.distribution_description_dim
        self.output_dim = self.messages_num * self.message_height * self.message_width

        self.output_mean_layer = nn.Sequential(
            nn.Linear(config.nn.n_h2, self.output_dim, dtype=torch.double),
            nn.Sigmoid(),  # 均值在0到1
        )

        self.output_std_layer = nn.Sequential(
            nn.Linear(config.nn.n_h2, self.output_dim, dtype=torch.double),
            nn.Sigmoid(),  # 主要是标准差要大于0，所以用了sigmoid
        )

    def forward(self, x):
        y = super(signaling_net, self).forward(x)
        mean = self.output_mean_layer(y)
        std = self.output_std_layer(y)

        messages_dist_descript_shape = torch.Size([x.shape[0]] + self.messages_shape.copy())
        mean_shaped = mean.view(messages_dist_descript_shape)
        std_shaped = std.view(messages_dist_descript_shape)

        # 只用gaussian
        batch_lmessages_dist = torch.distributions.Normal(mean_shaped, std_shaped)

        batch_messages_foreveryoneelse = batch_lmessages_dist.rsample(torch.Size([1]))
        shape = torch.Size([x.shape[0], self.messages_num, self.message_height, self.message_width])
        return batch_messages_foreveryoneelse.squeeze(dim=0).view(shape)
'''


# The signaling net should be stochastic.
# Its output is a continuous stochastic distribution, or PDF. The dim is inf.
# For now, the Gaussian distribution is used for testing. The outputs are the \mu and \sigma, for every pixel.
class signaling_net(net_base):
    def __init__(self, n_channels, config, agent_id=None, name='signaling_net_ind', chkpt_dir=default_dir, device=None):
        super(signaling_net, self).__init__(n_channels, config, agent_id=agent_id, name=name, chkpt_dir=chkpt_dir,
                                            device=device, howmany_linear=config.nn.signaling_net_howmany_linear)

        self.messages_num = config.env.n_agents - 1
        self.message_height = config.env.obs_height
        self.message_width = config.env.obs_width

        self.messages_shape = config.env.messages_shape[1:]  # 这个是乘了之后的
        self.distribution_description_dim = config.env.message_distribution_description_dim  # 目前只是2，只用gaussian
        # self.output_dim = self.messages_num \
        #                   * self.message_height * self.message_width * self.distribution_description_dim
        self.output_dim = self.messages_num * self.message_height * self.message_width

        self.output_mean_layer = nn.Sequential(
            nn.Linear(config.nn.n_h2, self.output_dim, dtype=torch.double),
            nn.Sigmoid(),  # 均值在0到1
        )

        self.output_std_layer = nn.Sequential(
            nn.Linear(config.nn.n_h2, self.output_dim, dtype=torch.double),
            nn.Sigmoid(),  # 主要是标准差要大于0，所以用了sigmoid
        )

    def forward(self, x):
        y = super(signaling_net, self).forward(x)
        mean = self.output_mean_layer(y)
        std = self.output_std_layer(y)

        mean_standard = torch.zeros_like(mean)
        std_standard = torch.ones_like(std)
        noise = torch.normal(mean_standard, std_standard)

        batch_messages_foreveryoneelse = mean + noise * std

        shape = torch.Size([x.shape[0], self.messages_num, self.message_height, self.message_width])
        return batch_messages_foreveryoneelse.view(shape)
