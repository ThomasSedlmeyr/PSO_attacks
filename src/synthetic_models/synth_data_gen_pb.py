import dill
import numpy as np
from synthesis.synthesizers.privbayes import PrivBayes

from src.synthetic_models.synthetic_base import SyntheticModel
from thomas.core import BayesianNetwork
from copy import copy


class SynthDataGenPB(SyntheticModel):
    def __init__(self, epsilon=1.0, verbose=False):
        super().__init__()
        self.synth_model = PrivBayes(epsilon=epsilon, verbose=verbose, n_cpus=None)

    def get_struct_enc(self):
        # get graph structure of model encoded as adjacency matrix
        network = self.synth_model.network_

        attrs = sorted(self.synth_model.columns_)
        n_attrs = len(attrs)
        adj_matrix = np.zeros((n_attrs, n_attrs))

        for index, pair in enumerate(network):
            to_idx = attrs.index(pair.attribute)
            if pair.parents:
                for parent in pair.parents:
                    from_idx = attrs.index(parent)
                    adj_matrix[to_idx, from_idx] = 1
        
        return adj_matrix.flatten().tolist()
    
    def get_raw_values(self):
        # get raw values stored in model

        self.synth_model.cpt_.values()
        probss = []
        #for col in self.metadata['columns']:
        #    probss.extend(self.synth_model.methods[col['name']].conditional_dist.flatten().tolist())
        for idx, pair in enumerate(self.synth_model.network_):
            probs = self.synth_model.cpt_[pair.attribute]
            probss.extend(probs.flat.tolist())
        return probss

    def fit(self, df):
        self.synth_model.fit(df)
    
    def sample(self, n_synth):
        return self.synth_model.sample(n_synth)

    def before_save(self):
        """
        Save this synthesizer instance to the given path using pickle.

        Parameters
        ----------
        path: str
            Path where the synthesizer instance is saved.
        """
        # todo issue can't save if model is fitted - likely error within thomas
        if hasattr(self.synth_model, 'model_'):
            pb = copy(self.synth_model)
            del pb.model_
            self.synth_model = pb
        else:
            pass

    def after_load(self):
        # recreate model_ attribute based on fitted network and cpt's
        if hasattr(self, 'cpt_'):
            self.synth_model.model_ = BayesianNetwork.from_CPTs('PrivBayes', self.cpt_.values())





