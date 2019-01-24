########################################
## Tests for utility.pyx
########################################

import unittest
import sys
import numpy as np
sys.path.append('belief_propagation_attack')

import utility

class MyTestCase(unittest.TestCase):
    def test_plaintext_array(self):
        p = utility.get_plaintext_array(1)
        self.assertTrue(utility.is_plaintext_array(p))

    def test_no_knowledge_array(self):
        no_knowledge = utility.get_no_knowledge_array()
        self.assertTrue(utility.is_no_knowledge_array(no_knowledge))

    def test_zeros_array(self):
        zero = utility.get_zeros_array()
        self.assertTrue(utility.is_zeros_array(zero))

    def test_xtimes(self):
        self.assertEquals(utility.xtimes(3), 6)
        self.assertEquals(utility.xtimes(211), 189)

    def test_inv_xtimes(self):
        self.assertEquals(utility.inv_xtimes(6), 3)
        self.assertEquals(utility.inv_xtimes(189), 211)

    def test_array_multiply(self):
        v1 = utility.get_zeros_array()
        v2 = utility.get_zeros_array()
        v1[0] = 0.5
        v2[0] = 1.0
        v2[1] = 1.0
        v_mult = utility.array_multiply(v1, v2)
        self.assertEquals(v1[0], 0.5)
        self.assertEquals(v2[0], 1.0)
        self.assertEquals(v2[1], 1.0)
        self.assertEquals(utility.array_max(v_mult), 1)
        self.assertEquals(v_mult[1], v_mult[2])

    def test_array_divide_float(self):
        v1 = utility.get_zeros_array()
        v1[0] = 10.0
        v1[1] = 5.0
        v1[2] = 1.0
        x = 2.5
        v_divide = utility.array_divide_float(v1, x)
        self.assertEquals(v_divide[0], 4.0)
        self.assertEquals(v_divide[1], 2.0)
        self.assertAlmostEquals(v_divide[2], 0.4)
        self.assertEquals(v_divide[3], 0.0)

    # def test_array_divide_float(self):

    def test_normalise_array(self):
        v1 = utility.get_filled_array(1)
        v_norm_v1 = utility.normalise_array(v1)
        self.assertTrue(utility.is_no_knowledge_array(v_norm_v1))
        self.assertAlmostEquals(np.sum(v_norm_v1), 1)

if __name__ == '__main__':
    unittest.main()
