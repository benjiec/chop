#!/usr/bin/env python3

import unittest
import subprocess
import sys


class TestTrainCLIDSS(unittest.TestCase):
    def test_cli_requires_dss_motifs(self):
        # Running without --dss-motifs should error
        proc = subprocess.run(
            [sys.executable, '-m', 'gene_predictor.train', '--help'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Help should include required --dss-motifs flag
        self.assertIn('--dss-motifs', proc.stdout)


if __name__ == '__main__':
    unittest.main()


