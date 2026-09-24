"""Regression tests for portable postprocessing, without historical data."""
import importlib.util
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parents[2]


def load(name,rel):
    spec = importlib.util.spec_from_file_location(name,CODE/rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DeliveryTests(unittest.TestCase):
    def test_one_sided_contrast_scales_once(self):
        mod = load('delivery_llm','overleaf/experiments/llm_cfpt/plot_real_llm_evidence.py')
        data = {'contrasts':{'calibration':{'Wstar-W0':{'KL':{'difference':-.06,'upper95':-.04}}},
                'test':{'Wstar-W0':{'G':{'difference':-.0005,'upper95':.0008}},
                        'Wstar-U0':{'Rnu':{'difference':.009,'upper95':.015}}}}}
        fig,ax = plt.subplots()
        try:
            mod.draw_intervention(ax,data)
            expected = [(-6.,-4.),(-.5,.8),(.9,1.5)]
            for container,(point,upper) in zip(ax.containers,expected):
                segment = container.lines[2][0].get_segments()[0]
                np.testing.assert_allclose(segment[:,0],[point,upper],atol=1e-14)
        finally:
            plt.close(fig)

    def test_undefined_reliability_is_preserved(self):
        mod = load('delivery_verify','tools/verify_historical.py')
        mod.close(float('nan'),float('nan'))
        mod.close(float('inf'),float('inf'))
        with self.assertRaises(AssertionError):
            mod.close(0.,float('nan'))

    def test_changed_numbers_are_rejected(self):
        mod = load('delivery_verify2','tools/verify_historical.py')
        with self.assertRaises(AssertionError):
            mod.close({'value':1.1},{'value':1.})


if __name__=='__main__':
    unittest.main()
