# Pommerman - RLlib

A wrapper for the [Pommerman](https://www.pommerman.com/) Multi-Agent Reinforcement Learning Environment
based on [Project Ray](https://github.com/ray-project/ray) - [RLlib](https://ray.readthedocs.io/en/latest/rllib.html)'s
`MultiAgentEnv`.    

## Requirements

**RLlib**:  
This wrapper requires a [`Ray RLlib`](https://ray.readthedocs.io/en/latest/rllib.html) 
release of at least [`0.6.1`](https://github.com/ray-project/ray/releases/tag/ray-0.6.0),
since this release allows for Dict observation spaces.  
**Pommerman**:    
Earlier releases than [Pommerman](https://github.com/MultiAgentLearning/playground)
`0.2.0` might work, but it's recommended to use `pommerman>=0.2.0`.    

## Usage
For a simple example of how to use this wrapped environment along with the custom 
preprocessor please refer to `train_example.py`. This example uses a policy gradient 
trainer to train four different policy gradient agents in one environment.  

This is work in progress so use with care.
