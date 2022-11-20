import torch

HR_action_maptoword = ['not hire', 'hire']
professor_action_maptoword = ['not recommend', 'recommend']
student_charc_maptoword = ['bad', 'good']
agent_typte_maptoword = ['professor', 'HR']

'''
        not hire  &  hire
bad     (0, 0)      (1, -1)
good    (0, 0)      (1, 1)
'''


class student_sampler(object):
    def __init__(self, prob_good):
        self.prob_good = prob_good
        self.prob_bad = 1 - prob_good
        self.prob_distribution = torch.tensor([self.prob_bad, self.prob_good], dtype=torch.double)
        self.sampler = torch.multinomial

    def sample(self, num=1):
        students_charac = self.sampler(self.prob_distribution, num, replacement=True)
        return students_charac


class recommendation_env(object):
    def __init__(self, config):
        self.config = config
        self.student_sampler = student_sampler(prob_good=self.config.prob_good)
        self.rewardmap_professor = config.rewardmap_professor
        self.rewardmap_HR = config.rewardmap_HR
        return

    def reset(self, ):
        student_charac = self.student_sampler.sample(self.config.num_sample)
        return student_charac

    def step(self, student_charac, hire_decision):
        student_charac = student_charac.detach().int()
        hire_decision = hire_decision.detach().int()
        reward_professor = self.rewardmap_professor[student_charac][hire_decision]
        reward_HR = self.rewardmap_HR[student_charac][hire_decision]
        return reward_professor * self.config.reward_magnification_factor, reward_HR * self.config.reward_magnification_factor
