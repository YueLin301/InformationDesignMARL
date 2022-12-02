import torch


class buffer_class(object):

    def __init__(self, ):
        self.reset()

    def reset(self):
        self.name = ['obs_sender', 'message', 'log_phi_sigma', 'ai', 'aj', 'log_pi_ait', 'log_pi_ajt', 'ri', 'rj']
        self.len_name = len(self.name)
        self.name_dict = dict(zip(self.name, range(self.len_name)))

        self.batch = [[]] * self.len_name

        return

    def add(self, transition):
        assert self.len_name == len(transition)
        for i in range(self.len_name):
            self.batch[i].append(transition[i])
        return


if __name__ == '__main__':
    buffer = buffer_class()

    batch_size = 99
    obs_height = 7
    obs_width = 7
    obs_depth = 5
    obs_sender = torch.rand(batch_size, obs_height, obs_width, obs_depth, dtype=torch.double)

    message_depth = 1
    message = torch.rand(batch_size, obs_height, obs_width, message_depth)

    # ai =
    print('haha')
