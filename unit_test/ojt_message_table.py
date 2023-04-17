import torch

if __name__ == '__main__':
    a = torch.arange(60).view(3, 5, 2, 2)

    b_raw = torch.arange(4)
    b = torch.nn.functional.one_hot(b_raw).view(4, 2, 2)

    b2 = b.unsqueeze(dim=1).unsqueeze(dim=0)
    b3 = b2.expand(3, -1, -1, -1, -1)
    a2 = a.unsqueeze(dim=1)
    a3 = a2.expand(-1, 4, -1, -1, -1)
    c = torch.cat([a3, b3], dim=2)

    c0 = c[0]
    c1 = c[1]
    c2 = c[2]

    print('done')
