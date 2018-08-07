"""
Unittest to set up GPU for an experiment of reproduce result in
Topological Signatures of Deep Learning.
"""

import unittest

import torch

class TestOneGPUSetting(unittest.TestCase):
    def test_enable_GPU(self):
        self.assertEqual(torch.cuda.is_available(), True)
    def test_torch_version(self):
        self.assertEqual(torch.__version__[:3], '0.3')
    def test_number_device(self):
        self.assertEqual(torch.cuda.device_count(), 1)
    def test_name_device(self):
        if torch.cuda.is_available():
            self.assertEqual(torch.cuda.get_device_name(0), 'Quadro K6000')

if __name__=='__main__':
    unittest.main()
