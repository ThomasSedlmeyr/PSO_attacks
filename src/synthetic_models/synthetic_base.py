import numpy as np
import dill

class SyntheticModel:
    """Base class for all generative models"""

    def __init__(self):
        """Initialize a generative model"""
        pass

    def fit(self, df):
        """Fit a generative model"""
        return NotImplementedError('Method needs to be overwritten.')

    def sample(self, n_synth):
        """Generate a synthetic dataset of size n_synth"""
        return NotImplementedError('Method needs to be overwritten.')
    
    def fit_and_sample(self, df, n_synth):
        """Fit and sample records in a single function"""
        self.fit(df)
        return self.sample(n_synth)
    

    def get_struct(self):
        """Get structure of synthetic data model (for white-box attacks)"""
        return None
    
    def get_struct_enc(self):
        """Get encoding of structure (for white-box attacks)"""
        return []
    
    def get_raw_values(self):
        """Get raw values from synthetic data model"""
        return []

    def load_model(self, path):
        """Restore fitted generative model"""
        model = dill.load(open(path, 'rb'))
        self.__dict__.update(model.__dict__)

    @classmethod
    def load(cls, path):
        """Restore fitted generative model"""
        loaded = dill.load(open(path, 'rb'))
        loaded.after_load()
        return loaded

    def save(self, save_path):
        """Save fitted generative model"""
        self.before_save()
        dill.dump(self, open(save_path, 'wb'))

    def get_wb_features(self, encode_graph_structure=True):
        feats = []
        if encode_graph_structure:
            feats = self.get_struct_enc()

        # calculate error in values extracted from synthetic model and corresponding values in the model w/o DP
        synth_vals = np.array(self.get_raw_values())
        feats = feats + synth_vals.tolist()
        return feats

    def before_save(self):
        pass

    def after_load(self):
        pass
