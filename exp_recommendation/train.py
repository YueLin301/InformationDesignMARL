from env import recommendation
from exp_recommendation.agent_class import pro_class, hr_class
from exp_recommendation.agent_formal_constrained import pro_formal_constrained
# from exp_recommendation.agent_class import hr_class
from exp_recommendation.agent_baseline import pro_baseline_class
from exp_recommendation.episode_generator import run_an_episode
from exp_recommendation.rec_utils import init_wandb, plot_with_wandb

from tqdm import tqdm


def set_Env_and_Agents(config, pro_type='regularized'):
    assert pro_type in ['baseline', 'formal_constrained', 'regularized']
    baseline = True if pro_type == 'baseline' else False

    print('----------------------------------------')
    print('Setting agents and env.')

    env = recommendation.recommendation_env(config.env)
    if pro_type == 'regularized':
        pro = pro_class(config)
    elif pro_type == 'baseline':
        pro = pro_baseline_class(config)
    else:
        pro = pro_formal_constrained(config)
    hr = hr_class(config)

    pro.build_connection(hr)
    hr.build_connection(pro)

    return env, pro, hr


def train(config, env, pro, hr, using_wandb=False, group_name=None, seed=None, pro_type='formal_constrained'):
    print('----------------------------------------')

    if using_wandb:
        alpha = config.pro.sender_objective_alpha if not pro_type == 'baseline' else config.pro.sender_objective_alpha - 1000
        chart_name_list, run_handle = init_wandb(group_name, alpha)
        if not seed is None:
            run_handle.tags = run_handle.tags + (str(seed),)
    fake_buffer = []  # for ploting curves (local)
    print('Start training.')

    pbar = tqdm(total=config.train.n_episodes)

    i_episode = 0
    while i_episode < config.train.n_episodes:
        buffer, fake_buffer = run_an_episode(env, pro, hr, fake_buffer)
        if using_wandb:
            plot_with_wandb(chart_name_list, buffer, i_episode, config.env.sample_n_students)
        i_episode += config.env.sample_n_students
        pbar.update(config.env.sample_n_students)

        hr.update_ac(buffer)
        pro.update_c(buffer)

        if not config.pro.fixed_signaling_scheme:
            buffer, fake_buffer = run_an_episode(env, pro, hr, fake_buffer)
            if using_wandb:
                plot_with_wandb(chart_name_list, buffer, i_episode, config.env.sample_n_students)
            pro.update_infor_design(buffer)
            i_episode += config.env.sample_n_students
            pbar.update(config.env.sample_n_students)

        # if not i_episode % 1e4:
        #     completion_rate = i_episode / config.train.n_episodes
        #     print('Task completion:\t{:.1%}'.format(completion_rate))

    if using_wandb:
        run_handle.finish()

    pbar.close()
    return fake_buffer
