import unittest
from pathlib import Path
import tempfile

from dna_learner.trainer import train as train_fn
from dna_learner.model import create_base_config
from utils.dataset import GenomicSyntheticTestingDataset, RandomBasesGenerator, RandomUTR5Generator, AddATGGenerator
from utils.sequences import KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES
from utils.constants import GenePredictionClass as P


class TestTrainerSamplerIntegration(unittest.TestCase):
    def test_trainer_runs_with_class_aware_sampler(self):
        # Build a tiny dataset with deterministic class presence in contigs
        utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        layouts = [
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
            RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.0),
            AddATGGenerator(),
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
        ]
        dataset = GenomicSyntheticTestingDataset(
            max_sequence_length=200,
            num_contigs=12,
            layouts_per_contig=1,
            layouts=layouts,
        )

        config = create_base_config(
            max_seq_length=200,
            num_classes=3,
            class_names=['INTERGENIC', 'UTR5', 'START'],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=3,
            class_weights=[1.0, 1.0, 1.0],
            attention_masks={0: 2},
            kmer_size=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model, val_loader = train_fn(
                dataset,
                config,
                Path(tmpdir),
                additional_callback_generator=lambda v: [],
                monitor_metric='val_loss',
                monitor_mode='min',
                batch_sampling=True,
            )
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(val_loader, '__iter__'))


if __name__ == '__main__':
    unittest.main()


