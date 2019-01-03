import atexit
import ray
from ray.tune.registry import register_env

from envs.v0 import Pomme_v0
from preprocessors.featurize_preprocessor import Featurize_Preprocessor
from pommerman.configs import team_competition_env

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.pg import PGAgent
from ray.rllib.agents.pg.pg_policy_graph import PGPolicyGraph

def shutdown():
    print('Shutting down Ray...')
    ray.shutdown()
atexit.register(shutdown)


if __name__ == '__main__':
    # register environment
    register_env('Pomme_v0', lambda config: Pomme_v0(config))
    # register preprocessor
    ModelCatalog.register_custom_preprocessor('Featurize_Preprocessor', Featurize_Preprocessor)

    ray.init()

    # get env config and create dummy instance to retrieve observation & action space
    env_config = team_competition_env()
    p = Pomme_v0(env_config)
    obs_space = p.pomme.observation_space
    act_space = p.pomme.action_space
    p.pomme.close()

    # initialize trainer - since all agent's use the same policy graph one trainer is fine here
    # otherwise we'd need one trainer per policy graph used for training
    trainer = PGAgent(env='Pomme_v0',
                      config={
                          'multiagent': {
                              'policy_graphs': {
                                  'agent_0': (PGPolicyGraph, obs_space, act_space, {"gamma": 0.85}),
                                  'agent_1': (PGPolicyGraph, obs_space, act_space, {"gamma": 0.90}),
                                  'agent_2': (PGPolicyGraph, obs_space, act_space, {"gamma": 0.95}),
                                  'agent_3': (PGPolicyGraph, obs_space, act_space, {"gamma": 0.99}),
                              },
                              'policy_mapping_fn': lambda agent_id: agent_id
                          },
                          'model': {
                              'custom_preprocessor': 'Featurize_Preprocessor'
                          },
                          'env_config': env_config
                      })
    print('\nTrainer Config:\n', trainer.config, '\n')

    # quick IPython embed to review the config
    import IPython
    IPython.embed()
    while True:
        print(trainer.train())
