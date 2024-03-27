from feature_extractor import FeatureExtractor
from contested_crossing import ContestedCrossing


class CCrossFeatureExtractor(FeatureExtractor):

    def __init__(self, mdp):
        self.mdp = mdp

    def num_features(self):
        return 4

    def num_actions(self):
        return len(self.mdp.get_actions())

    def extract(self, state, action):
        x = 0
        y = 1
        e = 0.01
        feature_values = []
        for a in self.mdp.get_actions():
            if a == action and state != ContestedCrossing.TERMINAL:
                feature_values += [(state[x] + e) / (self.mdp.width + e)]
                feature_values += [(state[y] + e) / (self.mdp.height + e)]
                feature_values += [
                    self.mdp.get_state_danger(state)
                ]
                feature_values += [(state[x]-self.mdp.battery[x]+
                                    state[y]-self.mdp.battery[y])/
                                   (self.mdp.width+self.mdp.height)]
            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]
        return feature_values

class CCrossSmallFeatureExtractor(FeatureExtractor):

    def __init__(self, mdp):
        self.mdp = mdp

    def num_features(self):
        return 3

    def num_actions(self):
        return len(self.mdp.get_actions())

    def extract(self, state, action):
        x = 0
        y = 1
        e = 0.01
        feature_values = []
        for a in self.mdp.get_actions():
            if a == action and state != ContestedCrossing.TERMINAL:
                feature_values += [(state[x] + e) / (self.mdp.width + e)]
                feature_values += [(state[y] + e) / (self.mdp.height + e)]
                feature_values += [(state[x]-self.mdp.battery[x]+
                                    state[y]-self.mdp.battery[y])/
                                   (self.mdp.width+self.mdp.height)]
            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]
        return feature_values
