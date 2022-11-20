import torch
import seaborn
from torch import autograd


class signaling_net_class(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2,
                            dtype=torch.double), torch.nn.ReLU(), )
        self.linear = torch.nn.Linear(3 * 5 * 5, 1 * 5 * 5, dtype=torch.double)

    def forward(self, obs):
        y1 = self.conv2d(obs).view(obs.shape[0], -1)
        message = self.linear(y1)
        shape = torch.Size([obs.shape[0], 1, 5, 5])
        return message.view(shape)


class signaling_net_class_gaussian(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2,
                            dtype=torch.double), torch.nn.ReLU(), )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * 5 * 5, 2 * 5 * 5, dtype=torch.double, ),
            torch.nn.ReLU()
        )

        self.message_shape = [1, 5 * 5]
        self.distribution_description_dim = 2

    def forward(self, obs):
        y1 = self.conv2d(obs).view(obs.shape[0], -1)
        y2 = self.mlp(y1)

        messages_shape = [self.distribution_description_dim, obs.shape[0]] + self.message_shape
        batch_messages_shape = torch.Size(messages_shape)
        messages_distribution_description = y2.view(batch_messages_shape)

        # So far the distribution is Gaussian.
        batch_messages_foreveryoneelse = torch.normal(mean=messages_distribution_description[0],
                                                      std=messages_distribution_description[1])

        shape = torch.Size([obs.shape[0], 1, 5, 5])
        return batch_messages_foreveryoneelse.view(shape)


class signaling_net_class_gaussian_reparameterization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2,
                            dtype=torch.double), torch.nn.ReLU(), )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * 5 * 5, 2 * 5 * 5, dtype=torch.double, ),
            torch.nn.Sigmoid()
        )

        self.message_shape = [1, 5 * 5]
        self.distribution_description_dim = 2

    def forward(self, obs):
        y1 = self.conv2d(obs).view(obs.shape[0], -1)
        y2 = self.mlp(y1)

        messages_shape = [self.distribution_description_dim, obs.shape[0]] + self.message_shape
        batch_messages_shape = torch.Size(messages_shape)
        messages_distribution_description = y2.view(batch_messages_shape)

        # So far the distribution is Gaussian.
        batch_lmessages_dist = torch.distributions.Normal(messages_distribution_description[0],
                                                          messages_distribution_description[1])

        batch_messages_foreveryoneelse = batch_lmessages_dist.rsample(torch.Size([1]))
        shape = torch.Size([obs.shape[0], 1, 5, 5])
        return batch_messages_foreveryoneelse.squeeze(dim=0).view(shape)


class mean_net_class(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2,
                            dtype=torch.double), torch.nn.ReLU(), )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * 5 * 5, 2 * 5 * 5, dtype=torch.double, ),
            torch.nn.Sigmoid()
        )

        self.message_shape = [1, 5 * 5]
        self.distribution_description_dim = 1

    def forward(self, obs):
        y1 = self.conv2d(obs).view(obs.shape[0], -1)
        y2 = self.mlp(y1)

        messages_shape = [self.distribution_description_dim, obs.shape[0]] + self.message_shape
        batch_messages_shape = torch.Size(messages_shape)
        messages_distribution_description = y2.view(batch_messages_shape)

        return messages_distribution_description


class std_net_class(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2,
                            dtype=torch.double), torch.nn.ReLU(), )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * 5 * 5, 2 * 5 * 5, dtype=torch.double, ),
            torch.nn.Sigmoid()
        )

        self.message_shape = [1, 5 * 5]
        self.distribution_description_dim = 1

    def forward(self, obs):
        y1 = self.conv2d(obs).view(obs.shape[0], -1)
        y2 = self.mlp(y1)

        messages_shape = [self.distribution_description_dim, obs.shape[0]] + self.message_shape
        batch_messages_shape = torch.Size(messages_shape)
        messages_distribution_description = y2.view(batch_messages_shape)

        return messages_distribution_description


class critic_class(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=7, out_channels=3, kernel_size=5, stride=1, padding=2,
                            dtype=torch.double), torch.nn.ReLU(), )
        self.linear = torch.nn.Linear(3 * 5 * 5, 1, dtype=torch.double)

    def forward(self, obs_and_message):
        y1 = self.conv2d(obs_and_message).view(obs_and_message.shape[0], -1)
        v = self.linear(y1)
        return v


def exp1():
    signaling_net = signaling_net_class()
    critic = critic_class()

    obs_sender = torch.rand(1, 6, 5, 5, dtype=torch.double)
    message = signaling_net(obs_sender)

    obs_receiver = torch.zeros(1, 6, 5, 5)
    obs_and_messsage_receiver = torch.cat([obs_receiver, message], dim=1)
    v = critic(obs_and_messsage_receiver)

    gradeta_v = autograd.grad(v, list(signaling_net.parameters()), retain_graph=True)

    return 1


def exp2():
    signaling_net = signaling_net_class()
    critic = critic_class()

    obs_sender = torch.rand(1, 6, 5, 5, dtype=torch.double)
    message = signaling_net(obs_sender)

    obs_receiver = torch.zeros(1, 6, 5, 5)
    obs_and_messsage_receiver = torch.cat([obs_receiver, message], dim=1)

    obs_and_messsage_receiver = obs_and_messsage_receiver[0]

    obs_receiver_new = torch.zeros(6, 5, 5)
    obs_and_messsage_receiver_new = torch.cat([obs_receiver_new, obs_and_messsage_receiver[6].unsqueeze(dim=0)], )

    v = critic(obs_and_messsage_receiver_new.unsqueeze(dim=0))

    gradeta_v = autograd.grad(v, list(signaling_net.parameters()), retain_graph=True)

    return


def exp3():
    signaling_net = signaling_net_class_gaussian()
    critic = critic_class()

    obs_sender = torch.rand(1, 6, 5, 5, dtype=torch.double)
    message = signaling_net(obs_sender)

    obs_receiver = torch.zeros(1, 6, 5, 5)
    obs_and_messsage_receiver = torch.cat([obs_receiver, message], dim=1)

    obs_and_messsage_receiver = obs_and_messsage_receiver[0]

    obs_receiver_new = torch.zeros(6, 5, 5)
    obs_and_messsage_receiver_new = torch.cat([obs_receiver_new, obs_and_messsage_receiver[6].unsqueeze(dim=0)], )

    v = critic(obs_and_messsage_receiver_new.unsqueeze(dim=0))

    gradeta_v = autograd.grad(v, list(signaling_net.parameters()), retain_graph=True)

    return


def exp4():
    # mean = torch.rand( 5, 5, dtype=torch.double)
    # std = torch.rand(5, 5, dtype=torch.double)
    mean = torch.tensor([0], dtype=torch.double)
    std = torch.tensor([1], dtype=torch.double)

    dist = torch.distributions.Normal(mean, std)
    samples = dist.rsample(torch.Size([100000]))
    seaborn.distplot(samples)
    # seaborn.histplot(samples)
    return


def exp5():
    # 说明这两个分布是独立的
    # mean = torch.rand( 5, 5, dtype=torch.double)
    # std = torch.rand(5, 5, dtype=torch.double)
    mean = torch.tensor([-1, 1], dtype=torch.double)
    std = torch.tensor([1, 1], dtype=torch.double)

    dist = torch.distributions.Normal(mean, std)
    samples = dist.rsample(torch.Size([100000]))

    samples1 = samples[:, 0]
    samples2 = samples[:, 1]
    # samples_new = torch.cat([samples1, samples2])
    seaborn.distplot(samples1)
    seaborn.distplot(samples2)
    return


def exp6():
    signaling_net = signaling_net_class_gaussian()
    critic = critic_class()

    obs_sender = torch.rand(1, 6, 5, 5, dtype=torch.double)
    message = signaling_net(obs_sender)

    obs_receiver = torch.zeros(1, 6, 5, 5)
    obs_and_messsage_receiver = torch.cat([obs_receiver, message], dim=1)

    obs_and_messsage_receiver = obs_and_messsage_receiver[0]

    obs_receiver_new = torch.zeros(6, 5, 5)
    obs_and_messsage_receiver_new = torch.cat([obs_receiver_new, obs_and_messsage_receiver[6].unsqueeze(dim=0)], )

    v = critic(obs_and_messsage_receiver_new.unsqueeze(dim=0))

    gradeta_v = autograd.grad(v, list(signaling_net.parameters()), retain_graph=True)
    return


def exp7():
    signaling_net = signaling_net_class_gaussian_reparameterization()
    critic = critic_class()

    obs_sender = torch.rand(1, 6, 5, 5, dtype=torch.double)
    message = signaling_net(obs_sender)

    obs_receiver = torch.zeros(1, 6, 5, 5)
    obs_and_messsage_receiver = torch.cat([obs_receiver, message], dim=1)

    obs_and_messsage_receiver = obs_and_messsage_receiver[0]

    obs_receiver_new = torch.zeros(6, 5, 5)
    obs_and_messsage_receiver_new = torch.cat([obs_receiver_new, obs_and_messsage_receiver[6].unsqueeze(dim=0)], )

    v = critic(obs_and_messsage_receiver_new.unsqueeze(dim=0))

    gradeta_v = autograd.grad(v, list(signaling_net.parameters()), retain_graph=True)

    return


# def exp8():
#     mean_net = mean_net_class()
#     std_net = std_net_class()
#     critic = critic_class()
#
#     obs_sender = torch.rand(1, 6, 5, 5, dtype=torch.double)
#     mean = mean_net(obs_sender)
#     std = std_net
#
#
#     obs_receiver = torch.zeros(1, 6, 5, 5)
#     obs_and_messsage_receiver = torch.cat([obs_receiver, message], dim=1)
#
#     obs_and_messsage_receiver = obs_and_messsage_receiver[0]
#
#     obs_receiver_new = torch.zeros(6, 5, 5)
#     obs_and_messsage_receiver_new = torch.cat([obs_receiver_new, obs_and_messsage_receiver[6].unsqueeze(dim=0)], )
#
#     v = critic(obs_and_messsage_receiver_new.unsqueeze(dim=0))
#
#     gradeta_v = autograd.grad(v, list(signaling_net.parameters()), retain_graph=True)
#
#     return


if __name__ == '__main__':
    exp7()
