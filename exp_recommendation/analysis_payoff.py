import torch
from exp_recommendation.configs import env_config

if __name__ == '__main__':
    p_pro0 = 0.1744
    p_pro1 = 0.3626
    phi_bad = torch.tensor([[p_pro0, 1 - p_pro0]], dtype=torch.float32)
    phi_good = torch.tensor([[p_pro1, 1 - p_pro1]], dtype=torch.float32)
    phi = torch.cat([phi_bad, phi_good])

    p_hr0 = 1
    p_hr1 = 0
    pi_not_rec = torch.tensor([[p_hr0, 1 - p_hr0]], dtype=torch.float32)
    pi_rec = torch.tensor([[p_hr1, 1 - p_hr1]], dtype=torch.float32)
    pi = torch.cat([pi_not_rec, pi_rec])

    prob_bad = torch.tensor(1 - env_config.config_env.prob_good, dtype=torch.float32)
    p_stu = torch.tensor([1 - env_config.config_env.prob_good, env_config.config_env.prob_good], dtype=torch.float32)

    p_stu_2x2 = p_stu.unsqueeze(dim=1).repeat(1, 2)
    p_and_phi = p_stu_2x2 * phi
    p_and_pi = torch.mm(p_and_phi, pi)

    rewardmap_pro_tensor = torch.tensor(env_config.config_env.rewardmap_professor, dtype=torch.float32)
    rewardmap_hr_tensor = torch.tensor(env_config.config_env.rewardmap_HR, dtype=torch.float32)

    reward_pro_table = p_and_pi * rewardmap_pro_tensor
    reward_hr_table = p_and_pi * rewardmap_hr_tensor

    reward_pro = torch.sum(reward_pro_table)
    reward_hr = torch.sum(reward_hr_table)

    print('reward_pro:{}, \treward_hr:{}'.format(reward_pro, reward_hr))
