import copy
from DIAL_origin.env import SwitchGame, DotDic
from DIAL_origin.agent import CNet, Agent
from DIAL_origin.config import opts
from DIAL_origin.train import Arena

if __name__ == '__main__':
    game = SwitchGame(DotDic(opts))
    cnet = CNet(opts)
    cnet_target = copy.deepcopy(cnet)
    agents = [None]
    for i in range(1, opts['game_nagents'] + 1):
        agents.append(Agent(DotDic(opts), game=game, model=cnet, target=cnet_target, agent_no=i))

    arena = Arena(DotDic(opts), game)
    arena.train(agents)
