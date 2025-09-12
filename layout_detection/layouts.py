import numpy as np
from utils.dataset import (
    GenomicSyntheticTestingDataset,
    RandomBasesGenerator,
    RandomUTR5Generator,
    AddATGGenerator,
)
from utils.sequences import KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES
from utils.constants import GenePredictionClass as P


def decoy_random_decopy_flanks():
    background_len = 450
    return [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len, target=P.INTERGENIC, decoy="ATG", max_decoy=3),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
    ]


def utr5_start_random_decoy_flanks(kozak_only=False, blind=False):
    background_len = 450
    if kozak_only:
        utr_choices = KOZAK_SEQUENCES
    else:
        utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.INTERGENIC if blind else P.UTR5, mutation_prob=0.1),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout


def utr5_spacer_start_random_decoy_flanks():
    background_len = 350
    utr_choices = UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.1).mask_ending_atgs(),
        RandomBasesGenerator(length=200, target=P.INTERGENIC, random_min_length=0, avoid="ATG"),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout


def blind_utr5_spacer_start_random_decoy_flanks():
    background_len = 350
    utr_choices = UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.INTERGENIC, mutation_prob=0.1).mask_ending_atgs(),
        RandomBasesGenerator(length=200, target=P.INTERGENIC, random_min_length=0, avoid="ATG"),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout


def generate_dataset(num_sequences: int, max_seq_length: int, layout_version: int, layouts_per_contig: int = 1):
    """Generate fresh test data aligned to the model's max sequence length."""
    print(f"Generating {num_sequences} test sequences...")

    layouts = []
    if layout_version == 1:
        layouts = [
            utr5_start_random_decoy_flanks(),
            decoy_random_decopy_flanks()
        ]
    elif layout_version == 2:
        layouts = [
            utr5_spacer_start_random_decoy_flanks(),
            utr5_start_random_decoy_flanks(kozak_only=True),
            decoy_random_decopy_flanks(),
            decoy_random_decopy_flanks()
        ]
    elif layout_version == 3:
        layouts = [
            blind_utr5_spacer_start_random_decoy_flanks(),
            utr5_start_random_decoy_flanks(kozak_only=True, blind=True),
            decoy_random_decopy_flanks(),
            decoy_random_decopy_flanks()
        ]
    else:
        assert False, "Unknown layout"

    dataset = GenomicSyntheticTestingDataset(
        max_sequence_length=max_seq_length,
        num_contigs=num_sequences,
        layouts_per_contig=layouts_per_contig,
        layouts=layouts,
    )

    # Sanity check 
    for contig_idx in range(dataset.num_contigs):
        full_sequence = dataset.contigs[contig_idx]
        full_targets = dataset.contig_targets[contig_idx]
        
        utr5_positions = np.sum(full_targets == 1)
        total_atgs = 0
        real_start_atgs = 0
        real_start_coords = []
        for i in range(len(full_sequence) - 2):
            if full_sequence[i:i+3] == 'ATG':
                total_atgs += 1
                if full_targets[i] == 2:  # Check if this ATG is labeled as START
                    if i > 0 and full_targets[i-1] != 2:
                        real_start_coords.append(i)
                    real_start_atgs += 1
        
        print(f"contig {contig_idx}: {real_start_atgs} real START ATGs ({real_start_coords}), {utr5_positions} UTR5 positions, {total_atgs} total ATGs, {len(full_sequence)} bps")
        if real_start_atgs > 1:
            print(full_targets)
        assert real_start_atgs == layouts_per_contig or real_start_atgs == 0

    return dataset
