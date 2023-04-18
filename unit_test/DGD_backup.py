# ''' BCE Obedience Constraint (Dual Gradient Descent) '''
# # T: buffer length, the amount of time-steps (s_t, oj_t)
# # M: the amount of signals
# # N: the amount of actions
# _, phi_sigma_st = self.send_message(obs_sender)  # (T,M)
#
# # ojt_sigma_table = torch.cat([obs_sender[:, 0:1, :, :], message], dim=1)
# ojt = obs_sender[:, 0:1, :, :]  # [ojt_0, message_0], [ojt_0, message_1], [ojt_0, message_2]
# ojt_repeat = ojt.unsqueeze(dim=1).expand(-1, self.message_num, -1, -1, -1)
# message_table = self.message_table_onehot.unsqueeze(dim=1).unsqueeze(dim=0).expand(ojt.shape[0], -1, -1, -1, -1)
# ojt_message_talbe = torch.cat([ojt_repeat, message_table], dim=2)
#
# shape_temp = [ojt.shape[0] * self.message_num] + list(ojt_message_talbe.shape[2:])
# ojt_message_reshape = ojt_message_talbe.reshape(shape_temp)  # (T*M)
#
# _, pi_a_sigma_table = self.receiver.choose_action(ojt_message_reshape)  # (T*M,N)