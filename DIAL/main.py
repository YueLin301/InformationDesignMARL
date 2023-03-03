import copy
from DIAL.env import SwitchGame, DotDic
from DIAL.agent import CNet, Agent
from DIAL.config import opts
from DIAL.train import Arena

if __name__ == '__main__':
    game = SwitchGame(DotDic(opts))
    cnet = CNet(opts)
    cnet_target = copy.deepcopy(cnet)
    agents = [None]
    for i in range(1, opts['game_nagents'] + 1):
        agents.append(Agent(DotDic(opts), game=game, model=cnet, target=cnet_target, agent_no=i))

    arena = Arena(DotDic(opts), game)
    arena.train(agents)
