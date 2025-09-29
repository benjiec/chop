import unittest
import numpy as np

from utils.constants import GenePredictionClass as P
from utils.genome import add_random_n_prefix


class DummyRng:
    def __init__(self, val: int):
        self._val = int(val)

    def randint(self, a: int, b: int) -> int:
        return self._val


class TestNPrefix(unittest.TestCase):
    def test_disabled_no_change(self):
        seq = 'ATGC'
        tgt = np.array([P.INTERGENIC]*4, dtype=np.int64)
        new_seq, new_tgt, pad = add_random_n_prefix(seq, tgt, enabled=False, rng=DummyRng(123))
        self.assertEqual(seq, new_seq)
        self.assertTrue(np.array_equal(tgt, new_tgt))
        self.assertEqual(pad, 0)

    def test_enabled_with_fixed_rng(self):
        seq = 'ATGC'
        tgt = np.array([P.INTERGENIC]*4, dtype=np.int64)
        new_seq, new_tgt, pad = add_random_n_prefix(seq, tgt, enabled=True, rng=DummyRng(5), min_len=5, max_len=5)
        self.assertEqual(pad, 5)
        self.assertTrue(new_seq.startswith('N'*5))
        self.assertEqual(new_seq[-4:], 'ATGC')
        self.assertEqual(len(new_tgt), 9)
        self.assertTrue(np.all(new_tgt[:5] == P.INTERGENIC))
        self.assertTrue(np.all(new_tgt[5:] == tgt))


if __name__ == '__main__':
    unittest.main()


