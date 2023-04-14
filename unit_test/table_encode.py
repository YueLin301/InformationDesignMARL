import torch


def unittest_each():
    dim = 5
    table = torch.rand(dim ** 2)

    new_table = []
    for i in range(dim):
        row = []
        for j in range(dim):
            idx = i * dim + j  # *****
            row.append(table[idx])
        new_table.append(row)

    new_table = torch.tensor(new_table)
    print(torch.isclose(table.view(new_table.shape), new_table))
    return


def unittest_batch():
    batch_size = 5
    dim = 2
    table = torch.randint(0, 100, (batch_size, dim ** 2,))

    idxi = torch.randint(0, dim, (batch_size,))
    idxj = torch.randint(0, dim, (batch_size,))
    idx = idxi * dim + idxj
    results = table[range(len(idx)), idx]

    return


def q_critic(s, *a):
    batch_size = 20
    dim = 2
    q_table = torch.randint(0, 100, (batch_size, dim ** 2))

    idx = a[0] * dim + a[1]
    q = q_table[range(len(idx)), idx]

    return q


def unittest_critic():
    dim=2
    ai = torch.randint(0, 2, (20,))
    aj = torch.randint(0, 2, (20,))

    results = q_critic(0, ai,aj)
    return 0


if __name__ == '__main__':
    # unittest_batch()
    unittest_critic()
