import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.create_shadow_datasets import apply_m
from src.analysis.perform_singling_out_attacks import get_X_Y_D_and_meta_data
from src.utils.utils import get_metadata, read_config
from src.data_processing.vulnerability_computation import compute_vulnerability, compute_distances_fast


class MyTestCase(unittest.TestCase):

    def test_if_replication_works(self):
        config = read_config()
        path = config.path_data + "adult/adult_cat.csv"
        df = pd.read_csv(path)
        meta_data = get_metadata(df)
        m1, _ = apply_m(df, meta_data, None, 1000, epsilon=1000, model_name="privbayes_dpart", seed=42)
        m2, _ = apply_m(df, meta_data, None, 1000, epsilon=1000, model_name="privbayes_dpart", seed=42)
        print("M1:")
        print(m1[0].head(5))
        print("M2:")
        print(m2[0].head(5))
        if not m1[0].equals(m2[0]):
            raise ValueError("Replication does not work")

    def test_multi_threaded(self):
        config = read_config()
        path = config.path_data + "adult/adult_cat.csv"
        df = pd.read_csv(path).head(100)
        vuln_single = compute_vulnerability(df, target_path=None, number_threads=None, show_progress=True, multi_processing=False)
        vuln_multi = compute_vulnerability(df, target_path=None, number_threads=22, show_progress=True)
        self.assertTrue(np.allclose(vuln_multi, vuln_single))
        print("Test")


if __name__ == '__main__':
    unittest.main()
