from configs import configfile_harvest_test

if __name__ == '__main__':


    config = configfile_harvest_test.get_config(wandb_config=False)

    env = Env(config.env)

    obs_list = env.reset()
    env.render()

    actions_list = [1, 2]
    obs_next_list, reward_list, done, info = env.step(actions_list)

    print('haha')