import random
import torch

name_cur = ['obs_sender', 'message', 'phi', 'obs_and_message_receiver', 'a', 'pi', 'ri', 'rj', ]
name_next = [item + '_next' for item in name_cur]
name = name_cur + name_next
len_name = len(name)
name_dict = dict(zip(name, range(len_name)))


class buffer_class(object):
    def __init__(self):
        self.name, self.len_name, self.name_dict = name, len_name, name_dict
        self.capacity = 50
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

        # if self.data_size > self.capacity:
        #     for i in range(len(self.data)):
        #         self.data[i].pop(0)
        #         self.data_size -= 1

    def sample_a_batch(self, batch_size, sample_the_last=False):
        assert batch_size <= self.data_size
        if not sample_the_last:
            idx = random.sample(list(range(self.data_size)), batch_size)
        else:
            idx = list(range(self.data_size - batch_size, self.data_size))

        batch_data = []
        for i in self.data:
            if not i[0] is None:
                item_tensor = torch.cat(i)
                batch_data.append(item_tensor[idx].clone())
            else:
                batch_data.append(None)

        batch = batch_class(batch_data)
        return batch


class batch_class(object):
    def __init__(self, batch_data):
        self.name, self.len_name, self.name_dict = name, len_name, name_dict
        self.data = batch_data
        self.batch_size = len(batch_data)
