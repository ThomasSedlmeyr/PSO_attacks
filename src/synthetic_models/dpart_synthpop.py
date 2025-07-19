"""
Class that encapsulates PrivBayes model from dpart repo
https://github.com/hazy/dpart
"""
import dill
import numpy as np
from dpart.engines import DPsynthpop

from src.synthetic_models.synthetic_base import SyntheticModel


def get_bounds(df):
    bounds = {}
    for col in df:
        col_data = df[col]
        if col_data.dtype.kind in "fui":
            bounds[col] = {"min": col_data.min(),
                           "max": col_data.max()}
        if col_data.dtype.name == "category":
            bounds[col] = {"categories": col_data.unique().to_list()}

    return bounds


class DPartSynthpop(SyntheticModel):
    def __init__(self, epsilon=None, bounds=None):
        self.synth_model = DPsynthpop(epsilon=epsilon)
        if bounds is not None:
            self.synth_model.bounds = bounds

    def get_struct(self):
        # get graph structure of model
        return self.synth_model.dep_manager.prediction_matrix
    
    def get_struct_enc(self):
        # get graph structure of model encoded as adjacency matrix
        network = self.get_struct()

        attrs = [col['name'] for col in self.metadata['columns']]
        n_attrs = len(attrs)
        adj_matrix = np.zeros((n_attrs, n_attrs))

        for to_idx, to_attr in enumerate(attrs):
            from_attrs = network[to_attr]
            for from_attr in from_attrs:
                from_idx = attrs.index(from_attr)
                adj_matrix[to_idx, from_idx] = 1
        
        return adj_matrix.flatten().tolist()
    
    def get_raw_values(self):
        # get raw values stored in model
        probss = []
        for col in self.metadata['columns']:
            probss.extend(self.synth_model.methods[col['name']].conditional_dist.flatten().tolist())
        return probss

    def fit(self, df):
        self.synth_model.fit(df)
    
    def sample(self, n_synth):
        return self.synth_model.sample(n_synth)

    def get_wb_features(self, encode_graph_structure=True):
        feats = []
        if encode_graph_structure:
            feats = self.get_struct_enc()

        # calculate error in values extracted from synthetic model and corresponding values in the model w/o DP
        synth_vals = np.array(self.get_raw_values())
        feats = feats + synth_vals.tolist()
        return feats
