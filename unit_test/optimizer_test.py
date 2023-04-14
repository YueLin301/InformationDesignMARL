import torch
from torch import autograd

if __name__ == '__main__':
    # w = torch.ones(1, dtype=torch.double, requires_grad=True)
    linear = torch.nn.Linear(1, 1, bias=False, dtype=torch.double)
    param = list(linear.parameters())
    print(param)

    optimizer = torch.optim.Adam(linear.parameters(), lr=1)

    optimizer.zero_grad()
    for par in linear.parameters():
        par.grad = torch.ones(1,1, dtype=torch.double)
    optimizer.step()

    param_after = list(linear.parameters())
    print(param_after)

    print('haha')
