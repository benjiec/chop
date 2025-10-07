import os
import tempfile

from utils.merge_split import (
    load_all_fasta,
    load_all_tsv,
    group_rows_by_sequence,
    pick_splits,
    merge_and_split,
    write_fasta,
    write_tsv,
)


def _write_tmp(path, content):
    with open(path, 'w') as f:
        f.write(content)


def test_merge_and_split_basic():
    with tempfile.TemporaryDirectory() as tmp:
        fna1 = os.path.join(tmp, 'a.fna')
        fna2 = os.path.join(tmp, 'b.fna')
        tsv1 = os.path.join(tmp, 'a.tsv')
        tsv2 = os.path.join(tmp, 'b.tsv')

        _write_tmp(fna1, ">s1\nATGC\n>s2\nATGC\n")
        _write_tmp(fna2, ">s3\nATGC\n>s4\nATGC\n")

        header = "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
        rows_a = "s1\tg1\t1\t4\t1\t4\t+\n"
        rows_b = "s3\tg2\t1\t4\t1\t4\t+\n"
        _write_tmp(tsv1, header + rows_a)
        _write_tmp(tsv2, header + rows_b)

        sequences, hdr, train_rows, test_rows, valid_sids = merge_and_split(
            tsv_inputs=[tsv1, tsv2],
            fasta_inputs=[fna1, fna2],
            num_train=1,
            num_test=1,
        )

        assert set(valid_sids) == {"s1", "s2", "s3", "s4"} & set(["s1", "s3"]) | set(["s2", "s4"])  # sequences only where TSV exists
        assert len(train_rows) == 1
        assert len(test_rows) == 1
        assert len(sequences) == 4

        # Write outputs to ensure IO works
        out_train_fna = os.path.join(tmp, 'train.fna')
        out_test_fna = os.path.join(tmp, 'test.fna')
        out_train_tsv = os.path.join(tmp, 'train.tsv')
        out_test_tsv = os.path.join(tmp, 'test.tsv')

        # extract ids from rows
        sid_idx = hdr.index('sequence_id')
        train_ids = list({r[sid_idx] for r in train_rows})
        test_ids = list({r[sid_idx] for r in test_rows})
        write_fasta(sequences, train_ids, out_train_fna)
        write_fasta(sequences, test_ids, out_test_fna)
        write_tsv(hdr, train_rows, out_train_tsv)
        write_tsv(hdr, test_rows, out_test_tsv)

        assert os.path.exists(out_train_fna)
        assert os.path.exists(out_test_fna)
        assert os.path.exists(out_train_tsv)
        assert os.path.exists(out_test_tsv)


def test_pick_splits_errors():
    try:
        pick_splits(["a"], num_train=-1, num_test=0)
        assert False, "Expected ValueError for negative num_train"
    except ValueError:
        pass
    try:
        pick_splits(["a"], num_train=1, num_test=1)
        assert False, "Expected ValueError for not enough sequences"
    except ValueError:
        pass


