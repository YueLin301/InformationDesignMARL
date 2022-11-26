import torch

if __name__ == '__main__':
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
    count_howmany = torch.sum(sample,dim=0)
    print('haha')
