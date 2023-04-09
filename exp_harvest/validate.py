from exp_harvest.configs.exp1_7 import config
from exp_harvest.agent_class import sender_class, receiver_class
from exp_harvest.episode_generator import run_an_episode
from exp_harvest.buffer_class import buffer_class
import torch

from env.harvest import Env

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    env = Env(config_env=config.env)
    sender = sender_class(config=config, device=device)
    receiver = receiver_class(config=config, device=device)

    sender.load_models()
    receiver.load_models()

    buffer = buffer_class()
    run_an_episode(env, sender, receiver, config, device, pls_render=True, buffer=buffer)

    batch = buffer.sample_a_batch(batch_size=config.train.batch_size)

    ri = batch.data[batch.name_dict['ri']]
    ri_sum = float(torch.sum(ri))
    rj = batch.data[batch.name_dict['rj']]
    rj_sum = float(torch.sum(rj))
    r_tot = ri + rj
    r_tot_sum = float(torch.sum(r_tot))

    print('ri:{}, rj:{}, sw:{}'.format(ri_sum, rj_sum, r_tot_sum))
    print('all done.')
