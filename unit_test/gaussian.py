import torch
import numpy as np
import matplotlib.pyplot as plt


def unittest1():
    mu = torch.tensor([1, 1], dtype=torch.double, requires_grad=True)
    sigma = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    dist = torch.distributions.Normal(mu, sigma)

    sample = dist.rsample([1000])
    x = np.array(sample[:, 0].detach())
    y = np.array(sample[:, 1].detach())
    plt.scatter(x, y, s=1)

    sample_round = torch.round(sample)
    x_round = np.array(sample_round[:, 0].detach())
    y_round = np.array(sample_round[:, 1].detach())
    plt.scatter(x_round, y_round, c='r')

    temp1 = torch.autograd.grad(sample[0, 0], mu, retain_graph=True)
    temp2 = torch.autograd.grad(sample[0, 0], sigma, retain_graph=True)

    plt.xlim([0, 2])
    plt.ylim([0, 2])
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
    sigma = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    dist = torch.distributions.Normal(mu, sigma)

    prob = torch.exp(dist.log_prob(torch.tensor([1, 2], dtype=torch.double))).unsqueeze(dim=0)
    # prob = dist.log_prob(torch.tensor([1, 2], dtype=torch.double))

    # probs = []
    # for i in range(5):
    #     probs_row = []
    #     for j in range(5):
    #         probs_row.append(torch.exp(dist.log_prob(torch.tensor([[i, j]], dtype=torch.double))).unsqueeze(dim=0))
    #     probs.append(torch.cat(probs_row))
    # probs = torch.cat(probs)

    return


def unittest5():
    mu = torch.tensor([1, 2], dtype=torch.double, requires_grad=True)
    sigma = torch.tensor([[0.3, 0],
                          [0, 0.3]], dtype=torch.double, requires_grad=True)
    dist = torch.distributions.MultivariateNormal(mu, sigma)

    # prob = torch.exp(dist.log_prob(torch.tensor([1, 2], dtype=torch.double))).unsqueeze(dim=0)

    # probs = []
    # for i in range(5):
    #     probs_row = []
    #     for j in range(5):
    #         probs_row.append(torch.exp(dist.log_prob(torch.tensor([[i, j]], dtype=torch.double))))
    #     probs.append(torch.cat(probs_row))
    # probs = torch.cat(probs).view(5,5)

    message_height = 5
    message_width = 5
    idxj = list(range(message_width))
    idx = [list(zip([i] * message_width, idxj)) for i in range(message_height)]
    idx_tensor = torch.tensor(idx)
    idx_tensor = idx_tensor.view(message_height * message_width, -1)

    temp = torch.exp(dist.log_prob(idx_tensor))

    return


if __name__ == '__main__':
    unittest1()
    print('all done.')
