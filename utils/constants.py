
ConventionalStopCodons = {'TAA', 'TAG', 'TGA'}


class GenePredictionClass:
    INTERGENIC = 0
    UTR5 = 1
    START = 2
    GENE = 3
    STOP = 4
    UTR3 = 5

    idx_to_cls = {
        INTERGENIC: 'INTERGENIC',
        UTR5: 'UTR5',
        START: 'START',
        GENE: 'GENE',
        STOP: 'STOP',
        UTR3: 'UTR3'
    }


class DNAEmbed:
    A = 0
    T = 1
    G = 2
    C = 3
    N = 4

    idx_to_bp = {
        A: 'A',
        T: 'T',
        G: 'G',
        C: 'C',
        N: 'N'
    }
