import numpy as np

from pommerman.envs.v0 import Pomme
from pommerman.configs import ffa_competition_env
from pommerman.configs import team_competition_env
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from ray.rllib.env.multi_agent_env import MultiAgentEnv


# constants
NUM_PLAYERS = 4
AGENT_IDS = ['agent_0', 'agent_1', 'agent_2', 'agent_3']

def to_dict(list):
    '''
    Turn list values into dictionary where keys are the agent IDs.
    '''
    return {AGENT_IDS[i]: list[i] for i in range(NUM_PLAYERS)}

def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1] * (3 - len(enemies))
    enemies = make_np_float(enemies)

    features = np.concatenate(
        (board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))
    return features  # np.expand_dims(features, 1)


class Pomme_v0(MultiAgentEnv):
    """
    An environment that hosts multiple independent agents.
    Agents are identified by (string) agent ids. Note that these "agents" here
    are not to be confused with RLlib agents.

    Examples:
    env = MyMultiAgentEnv()
    obs = env.reset()
    print(obs)
    {
        "car_0": [2.4, 1.6],
        "car_1": [3.4, -3.2],
        "traffic_light_1": [0, 3, 5, 1],
    }
    obs, rewards, dones, infos = env.step(
        action_dict={
            "car_0": 1, "car_1": 0, "traffic_light_1": 2,
        })
    print(rewards)
    {
        "car_0": 3,
        "car_1": -1,
        "traffic_light_1": 0,
    }
    print(dones)
    {
        "car_0": False,
        "car_1": True,
        "__all__": False,
    }
    """
    def __init__(self, config=team_competition_env()):
        self.pomme = Pomme(**config['env_kwargs'])
        self.agent_names = AGENT_IDS
        agent_list = []
        for i in range(4):
            agent_id = i
            agent_list.append(BaseAgent(config["agent"](agent_id, config["game_type"])))
        self.pomme.set_agents(agent_list)
        self.pomme.set_init_game_state(None)

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        obs_list = self.pomme.reset()
        return to_dict(obs_list)

    def step(self, action_dict):
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_name strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" is used to indicate env termination.
            infos (dict): Info values for each ready agent.
        """
        obs, rewards, done, info = self.pomme.step(list(action_dict.values()))
        dones = {'__all__': done}
        dones.update(to_dict([not agent.is_alive for agent in self.pomme._agents]))
        return to_dict(obs), to_dict(rewards), dones, info


if __name__ == '__main__':
    action_dict = {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}
    env = Pomme_v0()
    import IPython
    IPython.embed()
