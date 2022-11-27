class buffer_class(object):

    def __init__(self, ):
        self.reset()

    def reset(self):
        self.obs_pro, \
        self.message_onehot_pro, self.message_prob_pro, self.message_pro, \
        self.a_int_hr, self.a_prob_hr, self.a_logprob_hr, \
        self.reward_pro, self.reward_hr = [None] * 9

        self.values = None
        return

    def set_values(self, buffer_values):
        self.obs_pro, \
        self.message_onehot_pro, self.message_prob_pro, self.message_pro, \
        self.a_int_hr, self.a_prob_hr, self.a_logprob_hr, \
        self.reward_pro, self.reward_hr = buffer_values

        self.values = buffer_values
        return

    def get_values(self):
        return self.obs_pro, \
               self.message_onehot_pro, self.message_prob_pro, self.message_pro, \
               self.a_int_hr, self.a_prob_hr, self.a_logprob_hr, \
               self.reward_pro, self.reward_hr
