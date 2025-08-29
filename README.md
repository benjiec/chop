## Summary

This is a repository to develop a gene prediction program. The final end goal
is to take a genomic FASTA and predict list of protein amino acid sequences.

Currently we are focused on just figure out how to detect gene boundaries
within a genome.

### Fixture generation

The test data, or fixtures, should be genomic sequences with genes that include
5' UTR, START, CDS sequence, STOP, and 3' UTR, using common 5' and 3' UTRs
(e.g. Kozak, TATA, AAUAAA poly A signals, etc).

For now, for test fixture, leave 500 bps for 5' UTRs and 3' UTRS, and insert 5'
UTR sequences right before start codon and 3' UTR sequences after stop codon,
with some chances of mutation. Use biologically valid START (ATG) and STOP
codons (TAA/TAG/TGA).

For now, let's use max sequence length of 2500 for training the model. This
means for the test fixture, we are aiming to get gene length (exons and
introns) to be within 1200 bp or so, so we can add 1000 bps of UTRs to fit the
2500 bp sequence length. 10% of the test fixtures are long genes, so we can
test using sliding window to detect longer genes.

### Targets

We are marking START, STOP, and GENE BODY for bps between START and STOP. For
500 bps upstream of START, and 500 bps downstream of STOP, we are marking them
as possible UTRs. All other bps we are marking them INTERGENIC.

Let's not worry about overlapping genes. In the future, we can look at
predictions with less scores and use them to put together overlapping genes
(and alternative splicing).

### Running the end to end workflow

The following commands kick off an end to end workflow that a) generates test
fixtures - genomic FASTA and GFF, b) performs pre-processing to generate TSV
and gene context FASTA, c) trains model, d) predict genes, e) evaluates
model/prediction quality.

First, activate the virtual environment

```
cd /Users/benjie/git/chop && source chop_env/bin/activate
```

Then run the workflow:

```
python tests/gene_prediction/test_gene_prediction_workflow.py
```

## Current directory structure

chop/
├── scripts/
│   ├── preprocess_gene_data.py    # Stage 1: Preprocessing
├── training/
│   └── train_gene_prediction.py    # Stage 2 - Training
├── inference/
│   └── predict_gene_boundaries.py  # Stage 2 - Inference
├── tests/
│   ├── gene_prediction/                      # E2E test for stage 2
│   │   ├── test_gene_prediction_workflow.py  # Runner for E2E test for stage 2
│   │   ├── generate_test_fixture.py          # Generates test data for stage 2
│   │   ├── test_run_20241126_143052/     # Complete isolation per run
│   │   │   ├── fixtures/                 # Generated test data
│   │   │   ├── processed/                # Preprocessed data  
│   │   │   ├── models/                   # Trained models & checkpoints
│   │   │   └── predictions/              # Model predictions & results
│   └── unit/                        # Essential unit tests (25 tests)
├── utils/                          # Minimal utilities
├── models/                         # Model definitions
└── configs/
    └── gene_prediction_test.yaml   # Single config for stage 2 - gene prediction


## Cursor, please following these rules

1. When I ask you a question on why something is the way it is, do not assume I
want you to change it. Tell me if you want to change something and ask if that
is a change I want. Don't implement anything without a discussion. First
question should be "would you like me to propose a solution", rather than
"would you like me to implement". What would you implement w/o an agreed upon
approach?

2. Remember the purpose of a set of changes - and if that set of changes happen
to break something, the most important thing is NOT to fix what broke, but to
fix what broke AND preserve the purpose of the changes.

3. Don't change the configuration files and assume the code supports the syntax
changes in the configuration YAML files. Always check to make sure if you
add/modify the structure of the configuration YAML files, we have code that
supports the changes.

4. Do not produce python code to just print statements that you will display in
rich text in the chat box. Avoid using the terminal to communicate messages -
only use it to run commands.

5. Prefer concise answers to simple questions, rather than expanding on an
answer with possibilities and providing lots of text to read.

6. Run unit tests using `scripts/run-unit-tests` whenever you update something
not in the tests/ directory.

Whenever we are done implement a set of logic and validated that they are what
we want, we should add unit tests and update the tests/unit/run_tests.py script
to include the new unit tests.

7. Don't leave comments that only pertain to an action you took, e.g. to undo
something. Comments should describe the current state of the code or
configuration.

8. Always use chop_env virtual env.
