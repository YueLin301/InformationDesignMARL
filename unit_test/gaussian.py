import torch
import numpy as np
import matplotlib.pyplot as plt


def unittest1():
    mu = torch.tensor([1, 2], dtype=torch.double, requires_grad=True)
    sigma = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    dist = torch.distributions.Normal(mu, sigma)

    sample = dist.rsample([1000])
    x = np.array(sample[:, 0].detach())
    y = np.array(sample[:, 1].detach())
    plt.scatter(x, y)

    sample_round = torch.round(sample)
    x_round = np.array(sample_round[:, 0].detach())
    y_round = np.array(sample_round[:, 1].detach())
    plt.scatter(x_round, y_round, c='r')

    temp1 = torch.autograd.grad(sample[0, 0], mu, retain_graph=True)
    temp2 = torch.autograd.grad(sample[0, 0], sigma, retain_graph=True)

    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.show()

    return


def unittest2():
    mu = torch.tensor([1, 2], dtype=torch.double, requires_grad=True)
    sigma = torch.tensor([[0.5, -0.2],
                          [-0.2, 0.5]], dtype=torch.double, requires_grad=True)
    dist = torch.distributions.MultivariateNormal(mu, sigma)

    sample = dist.rsample([1000])
    x = np.array(sample[:, 0].detach())
    y = np.array(sample[:, 1].detach())
    plt.scatter(x, y)

    sample_round = torch.round(sample)
    x_round = np.array(sample_round[:, 0].detach())
    y_round = np.array(sample_round[:, 1].detach())
    plt.scatter(x_round, y_round, c='r')

    temp1 = torch.autograd.grad(sample[0, 0], mu, retain_graph=True)
    temp2 = torch.autograd.grad(sample[0, 0], sigma, retain_graph=True)

    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.show()

    return


def unittest3():
    dist = torch.distributions.Multinomial(total_count=1, probs=torch.tensor([0.2, 0.3, 0.5]))

    sample = dist.sample([1]).squeeze()
    log_prob_sample = dist.log_prob(sample)
    prob_sample = torch.exp(log_prob_sample)

    n_samples = dist.sample([1000])
    log_prob_samples = dist.log_prob(n_samples)
    prob_samples = torch.exp(log_prob_samples)

    return


def unittest4():
    mu = torch.tensor([1, 2], dtype=torch.double, requires_grad=True)
    sigma = torch.tensor([[0.5, -0.2],
                          [-0.2, 0.5]], dtype=torch.double, requires_grad=True)
    dist = torch.distributions.MultivariateNormal(mu, sigma)

    batch_size = 1000
    n_samples = dist.sample([batch_size])
    log_prob_samples = dist.log_prob(n_samples)
    prob_samples = torch.exp(log_prob_samples)

    return


if __name__ == '__main__':
    unittest1()
    # unittest2()
    # unittest3()
    # unittest4()
    print('all done.')
