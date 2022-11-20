class transition_class(object):

    def __init__(self, ):
        self.reset()

    def reset(self):
        self.obs_pro, \
        self.message_onehot_pro, self.message_prob_pro, self.message_pro, \
        self.a_int_hr, self.a_prob_hr, self.a_logprob_hr, \
        self.reward_pro, self.reward_hr = [None] * 9

        self.transition = None
        return

    def set_values(self, transition):
        self.obs_pro, \
        self.message_onehot_pro, self.message_prob_pro, self.message_pro, \
        self.a_int_hr, self.a_prob_hr, self.a_logprob_hr, \
        self.reward_pro, self.reward_hr = transition

        self.transition = transition
        return

    def get_values(self):
        return self.obs_pro, \
               self.message_onehot_pro, self.message_prob_pro, self.message_pro, \
               self.a_int_hr, self.a_prob_hr, self.a_logprob_hr, \
               self.reward_pro, self.reward_hr
