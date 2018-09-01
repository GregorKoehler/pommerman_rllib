from pommerman.envs.v0 import Pomme
from pommerman.configs import ffa_competition_env
from pommerman.configs import team_competition_env
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from ray.rllib.env.multi_agent_env import MultiAgentEnv


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
    def __init__(self, training_agent_id=None, seed=1, config=team_competition_env()):
        self.pomme = Pomme(**config['env_kwargs'])
        self.agent_names = ['agent_'+str(i) for i in range(4)]
        agent_list = []
        for i in range(4):
            agent_id = i
            agent_list.append(BaseAgent(config["agent"](agent_id, config["game_type"])))
        self.pomme.set_agents(agent_list)
        #self.pomme.set_training_agent(training_agent_id)
        self.pomme.seed(seed)
        self.pomme.set_init_game_state(None)

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        obs_list = self.pomme.reset()
        obs = {self.agent_names[i]: obs_list[i] for i in range(4)}
        return obs

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
        raise NotImplementedError

if __name__ == '__main__':

    env = Pomme_v0()
    import IPython
    IPython.embed()
