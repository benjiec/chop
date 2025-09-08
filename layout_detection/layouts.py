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


def utr5_start_random_decoy_flanks():
    background_len = 450
    utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.1),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout


def utr5_spacer_start_random_decoy_flanks():
    background_len = 400
    utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.1),
        RandomBasesGenerator(length=200, target=P.INTERGENIC, random_min_length=0, avoid="ATG"),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout


def blind_utr5_spacer_start_random_decoy_flanks():
    background_len = 400
    utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layout = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.INTERGENIC, mutation_prob=0.1),
        RandomBasesGenerator(length=200, target=P.INTERGENIC, random_min_length=0, avoid="ATG"),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]
    return layout
