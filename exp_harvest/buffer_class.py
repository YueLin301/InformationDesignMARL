class buffer_class(object):
    def __init__(self, ):
        self.reset()

    def reset(self):
        self.name = ['obs_sender', 'message', 'phi', 'obs_and_message_receiver', 'ai', 'pii', 'aj', 'pij', 'ri', 'rj']
        self.len_name = len(self.name)
        self.name_dict = dict(zip(self.name, range(self.len_name)))

        self.batch = [[], [], [], [], [], [], [], [], [], []]

    def add(self, transition):
        assert self.len_name == len(transition)
        for i in range(self.len_name):
            self.batch[i].append(transition[i])
