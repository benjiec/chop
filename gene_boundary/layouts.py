import random
import numpy as np
from utils.dataset import (
    GenomicSyntheticTestingDataset,
    RandomChoiceGenerator,
    RandomBasesGenerator,
    RandomUTR5Generator,
    AddATGGenerator,
    AddStopGenerator
)
from utils.sequences import KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES, UTR3_REAL_SEQUENCES, UTR3_COMPLETE_SEQUENCES
from utils.constants import GenePredictionClass as P, ConventionalStopCodons


def decoy_random_decopy_flanks():
    background_len = 1000
    return [
        RandomBasesGenerator(length=background_len, target=P.INTERGENIC, random_min_length=background_len // 2),
    ]


def blind_kozak_start_random_decoy_flanks():
    background_len = 400
    utr_choices = KOZAK_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.INTERGENIC, mutation_prob=0.05),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout


def blind_utr5_spacer_start_random_decoy_flanks():
    background_len = 350
    utr_choices = UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.INTERGENIC, mutation_prob=0.05).mask_ending_atgs(),
        RandomBasesGenerator(length=200, target=P.INTERGENIC, random_min_length=0, avoid="ATG"),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout


def blind_stop_utr3_random_decoy_flanks():
    background_len = 350
    utr_choices = UTR3_REAL_SEQUENCES + UTR3_COMPLETE_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, random_min_length=background_len // 4, avoid=ConventionalStopCodons),
        AddStopGenerator(),
        RandomChoiceGenerator(choices=utr_choices, target=P.INTERGENIC, mutation_prob=0.05),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy=ConventionalStopCodons, max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout


def blind_stop_spacer_utr3_random_decoy_flanks():
    background_len = 350
    utr_choices = UTR3_REAL_SEQUENCES + UTR3_COMPLETE_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, random_min_length=background_len // 4, avoid=ConventionalStopCodons),
        AddStopGenerator(),
        RandomBasesGenerator(length=100, target=P.INTERGENIC, random_min_length=0),
        RandomChoiceGenerator(choices=utr_choices, target=P.INTERGENIC, mutation_prob=0.05),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy=ConventionalStopCodons, max_decoy=3, random_min_length=background_len // 4)
    ]
    return layout


def blind_full_gene():
    background_len = 200
    utr5_choices = UTR5_REAL_SEQUENCES + IRES_SEQUENCES + KOZAK_SEQUENCES
    utr3_choices = UTR3_REAL_SEQUENCES + UTR3_COMPLETE_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr5_choices, target=P.INTERGENIC, mutation_prob=0.05).mask_ending_atgs(),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len * 2, target=P.GENE, random_min_length=background_len),
        RandomBasesGenerator(length=background_len // 2, target=P.GENE, random_min_length=background_len // 4, avoid=ConventionalStopCodons),
        AddStopGenerator(),
        RandomBasesGenerator(length=100, target=P.INTERGENIC, random_min_length=0),
        RandomChoiceGenerator(choices=utr3_choices, target=P.INTERGENIC, mutation_prob=0.05),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 2),
    ]
    return layout


def generate_dataset(num_sequences: int, max_seq_length: int, layouts_per_contig: int = 1, add_negatives = False, incl_start = True, incl_stop = True):
    print(f"Generating {num_sequences} test sequences...")

    layouts = []

    if incl_start:
        layouts.extend([
            blind_utr5_spacer_start_random_decoy_flanks(),
            blind_kozak_start_random_decoy_flanks()
        ])
        if add_negatives:
            layouts.append(decoy_random_decopy_flanks())

    if incl_stop:
        layouts.extend([
            blind_stop_utr3_random_decoy_flanks(),
            blind_stop_spacer_utr3_random_decoy_flanks()
        ])
        if add_negatives:
            layouts.append(decoy_random_decopy_flanks())

    if incl_start and incl_stop:
        layouts.extend([
            blind_full_gene(),
            blind_full_gene()
        ])

    random.shuffle(layouts)

    dataset = GenomicSyntheticTestingDataset(
        max_sequence_length=max_seq_length,
        num_contigs=num_sequences,
        layouts_per_contig=layouts_per_contig,
        layouts=layouts,
    )

    for contig_idx in range(dataset.num_contigs):
        full_sequence = dataset.contigs[contig_idx]
        full_targets = dataset.contig_targets[contig_idx]
        stop_targets = len([x for x in full_targets if x == P.STOP])
        start_targets = len([x for x in full_targets if x == P.START])
        if contig_idx % 100 == 0:
            print(f"contig {contig_idx}: {len(full_sequence)} bps, {stop_targets} STOP bps, {start_targets} START bps")
        assert stop_targets in (0, 3) and start_targets in (0, 3)

    return dataset
