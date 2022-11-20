import torch

# ========================================
# q(s,a), rather than q(s,sigma)
critic_pro = torch.nn.Sequential(
    torch.nn.Linear(in_features=4, out_features=4, bias=False, dtype=torch.double), torch.nn.ReLU(),
    torch.nn.Linear(in_features=4, out_features=1, bias=False, dtype=torch.double)
)

signaling_net = torch.nn.Sequential(
    # input: one hot; output: signaling 0/1 prob. distribution
    torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.double),
    torch.nn.Softmax(dim=-1)
)

# ========================================

critic_hr = torch.nn.Sequential(
    torch.nn.Linear(in_features=4, out_features=4, bias=False, dtype=torch.double), torch.nn.ReLU(),
    torch.nn.Linear(in_features=4, out_features=1, bias=False, dtype=torch.double)
)

actor = torch.nn.Sequential(
    torch.nn.Linear(in_features=2, out_features=2, bias=False, dtype=torch.double),
    torch.nn.Softmax(dim=-1)
)
