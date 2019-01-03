from pommerman.envs.v0 import Pomme
from pommerman import configs as pommerman_cfg
from pommerman import agents
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# constants
NUM_PLAYERS = 4
AGENT_IDS = ['agent_0', 'agent_1', 'agent_2', 'agent_3']

def to_dict(list):
    '''
    Turn list values into dictionary where keys are the agent IDs.
    '''
    return {AGENT_IDS[i]: list[i] for i in range(NUM_PLAYERS)}


class Pomme_v0(MultiAgentEnv):
    '''
    A wrapped Pommerman v0 environment for usage with Ray RLlib. The v0 environment is the base environment used in
    the NIPS'18 competition. Contrary to v1 it doesn't collapse walls and also doesn't allow for radio communication
    between agents (as does v2).

    Agents are identified by (string) agent IDs: `AGENT_IDS`
    (Note that these "agents" here are not to be confused with RLlib agents.)
    '''
    def __init__(self, config=pommerman_cfg.team_competition_env()):
        '''
        Initializes the Pommerman environment and adds Dummy Agents as expected by `Pomme`.

        Args:
            config (dict): A config defining the game mode. Options include FFA mode, team (2v2) and team radio (2v2).
            See pommerman's config.py and docs for more details.
        '''
        self.pomme = Pomme(**config['env_kwargs'])
        self.observation_space = dict
        self.action_space = self.pomme.action_space
        self.agent_names = AGENT_IDS
        agent_list = []
        for i in range(4):
            agent_id = i
            agent_list.append(agents.BaseAgent(config["agent"](agent_id, config["game_type"])))
        self.pomme.set_agents(agent_list)
        self.pomme.set_init_game_state(None)

    def reset(self):
        """
        Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        obs_list = self.pomme.reset()
        #return {key: featurize(val) for key, val in to_dict(obs_list).items()}
        return {key: val for key, val in to_dict(obs_list).items()}

    def step(self, action_dict):
        """
        Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The number of agents in the env can vary over
        time.

        Returns:
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the episode is just started, the value will be zero.
            dones (dict): Done values for each ready agent. The key "__all__" is used to indicate the end of the game.
            infos (dict): Info values for each ready agent.
        """
        # default actions since Pommerman env expects actions even if agent is dead
        actions = {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}
        # update actions with the ones returned from the policies
        actions.update(action_dict)
        # perform env step (expects a list)
        obs, rewards, done, info = self.pomme.step(list(actions.values()))
        # build 'dones' dictionary, key __all__ indicates env termination
        dones = {'__all__': done}
        # fetch all
        done_agents = to_dict([not agent.is_alive for agent in self.pomme._agents])
        # filter done dictionary to only return agents which are still alive
        # -> apparently this is how rllib determines when agents "die"
        dones.update({key: val for key, val in done_agents.items() if not val})
        # turn info dict into dictionary with agent IDs as keys
        infos = {AGENT_IDS[i]:{info_k: info_v for info_k, info_v in info.items()} for i in range(NUM_PLAYERS)}
        return to_dict(obs), to_dict(rewards), dones, infos


if __name__ == '__main__':
    action_dict = {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}
    env = Pomme_v0()
    import IPython
    IPython.embed()
