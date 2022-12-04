import random

name_cur = ['obs_sender', 'message', 'phi', 'obs_and_message_receiver',
            'ai', 'pii', 'aj', 'pij', 'ri', 'rj', ]
name_next = [item + '_next' for item in name_cur]
name = name_cur + name_next
len_name = len(name)
name_dict = dict(zip(name, range(len_name)))


class buffer_class(object):
    def __init__(self):
        self.name, self.len_name, self.name_dict = name, len_name, name_dict
        self.reset()

    def reset(self):
        self.data = [[] for _ in range(self.len_name)]
        assert len(self.data) == self.len_name, \
            'len(batch) should be {}, but now it is {}.'.format(self.len_name, len(self.data))
        self.data_size = 0

    def add_half_transition(self, half_transition, which_half):
        assert self.len_name // 2 == len(half_transition)
        assert which_half in ['1st', '2nd']

        if which_half == '1st':
            for i in range(self.len_name // 2):
                self.data[i].append(half_transition[i])
        else:
            for i in range(self.len_name // 2, self.len_name):
                self.data[i].append(half_transition[i - self.len_name // 2])
            self.data_size += 1

    def sample_a_batch(self, batch_size):
        assert batch_size <= self.data_size
        batch_data = []
        for i in self.data:
            batch_data.append(random.sample(i, batch_size))

        batch = batch_class(batch_data)
        return batch


class batch_class(object):
    def __init__(self, batch_data):
        self.name, self.len_name, self.name_dict = name, len_name, name_dict
        self.data = batch_data
        self.batch_size = len(batch_data)
