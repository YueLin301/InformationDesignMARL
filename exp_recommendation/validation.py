'''
The following cross entropy measures the degree of dishonesty:
    H(phi(~|1), phi(~|0))
    = - phi(0|1) * log2 phi(0|0) - phi(1|1) * log2 phi(1|0).
'''

'''
row: [0, 0.15, 0.3, 0.45, 0.6]
column: [0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10]
'''

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def calculate_cross_entropy(p, q):
    return torch.sum(- torch.log(q) * p)


def generate_heatmap_plt(data):
    fig = plt.figure(dpi=300)

    p = phi_when_bad = data[:, :, 0]
    q = phi_when_good = data[:, :, 1]
    heatmap = - p * torch.log(q) - (1 - p) * torch.log(1 - q)
    plt.imshow(heatmap, )

    plt.xticks([2.5 * i for i in range(0, 5)])
    plt.yticks([0.15 * i for i in range(0, 5)])
    plt.show()
    return


def generate_heatmap(data):
    fig = plt.figure(dpi=300)

    p = phi_when_bad = data[:, :, 0]
    q = phi_when_good = data[:, :, 1]
    heatmap = - p * torch.log(q) - (1 - p) * torch.log(1 - q)

    df = pd.DataFrame(heatmap,
                      # index=[0.15 * i for i in range(0, 5)],
                      index=[0.0, 0.15, 0.3, 0.45, 0.6],
                      columns=[2.5 * i for i in range(0, 5)])
    ax = sns.heatmap(data=df, cmap='crest', annot=True, vmin=0, vmax=10)
    ax.set(xlabel="lambda", ylabel="epsilon")
    plt.show()

    return


def generate_heatmap_fromfile(file_location0, file_location1):
    data_phi_0 = pd.read_csv(file_location0, )
    data_phi_1 = pd.read_csv(file_location1, )

    p = torch.tensor(data_phi_0.values)
    q = torch.tensor(data_phi_1.values)

    m = 1 / 2 * (p + q)
    JS = 1 / 2 * (
            p * (torch.log(p) - torch.log(m)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - m))
    ) + \
         1 / 2 * (
                 q * (torch.log(q) - torch.log(m)) + (1 - q) * (torch.log(1 - q) - torch.log(1 - m))
         )

    # h = - p * torch.log(q) - (1 - p) * torch.log(1 - q)
    # h = - q * torch.log(p) - (1 - q) * torch.log(1 - p)
    # h = q - p
    h_df = pd.DataFrame(JS,
                        index=[0.0, 0.15, 0.3, 0.45, 0.6],
                        columns=data_phi_0.columns)

    fig = plt.figure(dpi=300)
    ax = sns.heatmap(data=h_df, cmap='crest', annot=False, square=True)
    ax.set(xlabel="lambda", ylabel="epsilon")
    plt.show()

    return


if __name__ == '__main__':
    # generate_heatmap(data)
    generate_heatmap_fromfile('./data_results/data_phi_0.csv', './data_results/data_phi_1.csv')
    print('all done.')
