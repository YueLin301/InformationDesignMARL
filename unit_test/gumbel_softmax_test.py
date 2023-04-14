import torch


def unittest_normal_softmax():
    logits = torch.ones(98, 2)
    distributions = torch.softmax(logits, dim=-1)
    return


def unittest_recommendation():
    # temperature = 100
    # temperature = 1
    temperature = 0.5
    # temperature = 0.01
    # temperature = 0.001
    # p = 1 / 3
    p = 0.1
    # p = 0
    distribution = torch.tensor([p, 1 - p], dtype=torch.double)
    distribution = torch.log(distribution)
    distribution = distribution.unsqueeze(dim=0).repeat(1000, 1)
    # sample = torch.nn.functional.gumbel_softmax(distribution, tau=temperature, hard=True)
    # sample = torch.nn.functional.gumbel_softmax(distribution, tau=temperature, hard=True)
    sample = torch.nn.functional.gumbel_softmax(distribution, tau=temperature, hard=False)

    message_table = torch.tensor([0, 1], dtype=torch.double)
    message = torch.einsum('i,ji->j', message_table, sample)

    count_howmany1 = torch.sum(message)
    count_howmany = torch.sum(sample, dim=0)
    print('haha')
    return


def unittest_harvest1():
    temperature = 0.01

    p = torch.rand(7, 7, dtype=torch.double)
    p_flatten = torch.flatten(p).unsqueeze(dim=0)
    distribution = torch.cat([p_flatten, 1 - p_flatten])

    distribution = torch.log(distribution)
    distribution_transposed = torch.transpose(distribution, 0, 1)
    sample = torch.nn.functional.gumbel_softmax(distribution_transposed, tau=temperature, hard=True)

    message_table = torch.tensor([0, 1], dtype=torch.double)
    message_flatten = torch.einsum('i,ji->j', message_table, sample)

    message = message_flatten.view(7, 7)
    return


def unittest_harvest2():
    temperature = 0.01

    batch_size = 1000

    p = torch.rand(7, 7, 2, dtype=torch.double)
    p_flatten = torch.flatten(p).unsqueeze(dim=0)
    distribution = torch.cat([1 - p_flatten, p_flatten])

    distribution = torch.log(distribution)
    distribution_transposed = torch.transpose(distribution, 0, 1)
    batch_distribution_transposed = distribution_transposed.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    sample = torch.nn.functional.gumbel_softmax(batch_distribution_transposed, tau=temperature, hard=True)

    message_table = torch.tensor([0, 1], dtype=torch.double)
    message_flatten = torch.einsum('i,kji->kj', message_table, sample)

    message = message_flatten.view(batch_size, 7, 7, 2)

    count_howmany = torch.sum(message, dim=0)
    return


if __name__ == '__main__':
    # unittest_harvest2()
    unittest_normal_softmax()
