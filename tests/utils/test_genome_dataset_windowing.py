import unittest
import tempfile
from pathlib import Path
import gzip

import numpy as np

from utils.genome import AnnotatedGenomeDataset
from utils.windowing import compute_window_slices


def write_temp(path: Path, content: str):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt') as f:
            f.write(content)
    else:
        with open(path, 'w') as f:
            f.write(content)


class TestAnnotatedGenomeDatasetWindowing(unittest.TestCase):
    def test_windowing_splits_contig(self):
        # Simple contig of length 50 with a single + strand exon [10..40] (1-based)
        # Ensure START (ATG) at exon start and STOP (TAA) at exon end per AnnotatedGenomeDataset invariants
        seq_list = list('N' * 50)
        # 0-based positions: start_pos = 9..11, stop_pos = 37..39
        seq_list[9:12] = list('ATG')
        seq_list[37:40] = list('TAA')
        fasta = f">ctg\n{''.join(seq_list)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "ctg\tg\t10\t40\t10\t40\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "s.fna.gz"
            tsv_path = fp / "a.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)

            # Window smaller than contig -> expect multiple items
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), window=16, stride=8, random_prefix_ns=False)
            expected = compute_window_slices(50, window=16, stride=8)
            self.assertEqual(len(ds), len(expected))

            # Check each window length and that encoding/targets match lengths
            for i, (s, e) in enumerate(expected):
                x, y = ds[i]
                self.assertEqual(x.shape[0], e - s)
                self.assertEqual(y.shape[0], e - s)

            # No windowing -> single item of full length
            ds_full = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), random_prefix_ns=False)
            self.assertEqual(len(ds_full), 1)
            x_full, y_full = ds_full[0]
            self.assertEqual(x_full.shape[0], 50)
            self.assertEqual(y_full.shape[0], 50)

    def test_windowing_pads_last_short_window_to_fixed_length(self):
        # Contig length 30; use window 16 and stride 16 so second window is only length 14 before padding
        seq_list = list('N' * 30)
        # Add valid START/STOP so targets build without None
        seq_list[0:3] = list('ATG')
        seq_list[27:30] = list('TAA')
        fasta = f">ctg\n{''.join(seq_list)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "ctg\tg\t1\t30\t1\t30\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "s.fna.gz"
            tsv_path = fp / "a.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)

            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), window=16, stride=16, random_prefix_ns=False)
            # Expect two windows
            self.assertEqual(len(ds), 2)
            # First is full 16, second should be padded to 16
            x0, y0 = ds[0]
            x1, y1 = ds[1]
            self.assertEqual(x0.shape[0], 16)
            self.assertEqual(y0.shape[0], 16)
            self.assertEqual(x1.shape[0], 16)
            self.assertEqual(y1.shape[0], 16)
