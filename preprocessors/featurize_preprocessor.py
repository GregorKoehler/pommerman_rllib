import numpy as np
from ray.rllib.models.preprocessors import Preprocessor


class Featurize_Preprocessor(Preprocessor):
    '''
    Observation preprocessor turning dict-like Pommerman observations into a numpy vector
    of shape (372,).

    Attributes:
        shape (obj): Shape of the preprocessed output.
    '''
    def __init__(self, obs_space, options=None):
        super().__init__(obs_space, options)

    def _init_shape(self, obs_space, options):
        '''
        Returns the shape after preprocessing.

        For now returns hard-coded shape for featurized observations.
        '''
        return (372,)

    def transform(self, observation):
        """Returns the preprocessed observation."""
        return self._featurize(observation)

    def _featurize(self, obs):
        '''
        Turn board observations (dict) into a feature vector (numpy array).
        '''
        board = obs["board"].reshape(-1).astype(np.float32)
        bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
        bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
        position = self._make_np_float(obs["position"])
        ammo = self._make_np_float([obs["ammo"]])
        blast_strength = self._make_np_float([obs["blast_strength"]])
        can_kick = self._make_np_float([obs["can_kick"]])

        teammate = obs["teammate"]
        if teammate is not None:
            teammate = teammate.value
        else:
            teammate = -1
        teammate = self._make_np_float([teammate])

        enemies = obs["enemies"]
        enemies = [e.value for e in enemies]
        if len(enemies) < 3:
            enemies = enemies + [-1] * (3 - len(enemies))
        enemies = self._make_np_float(enemies)

        features = np.concatenate(
            (board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))
        return features

    def _make_np_float(self, feature):
        '''
        Return given feature as numpy array.
        '''
        return np.array(feature).astype(np.float32)
