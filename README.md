## Summary

This is a repository to develop a gene prediction program. The final end goal
is to take a genomic FASTA and a genomic GFF file as input, and predict list of
protein amino acid sequences.

The strategy we are taking is to develop two transformer NN models, with
attentions. The first model is a gene boundary prediction model that predicts
gene boundaries, specifically the START codon at the start of the first exon
and the STOP codon at the end of the last exon. The second model is a splicing
prediction model that predicts where exons are between START and STOP.

We are going to implement this strategy in four stages.

Stage 1: pre-processing of genomic fasta and GFF into a TSV of annotations and
a fasta of gene context sequences. The TSV of annotations includes gene ID,
gene start, gene end, exon ID, exon start, and exon end. The coordinates are
adjusted to the gene context sequences, each is ~2000 bps flanking the START
and STOP codons. The gene context sequences are also strand normalized and are
always specified as 5' to 3' + strand sequences.

Stage 2: training of gene boundary prediction model.

Stage 3: training of splicing prediction model.

Stage 4: Convert the gene and splicing prediction results and generate a new
GFF and an amino acid fasta file.


## Gene boundary prediction model development

To develop this model, let's first develop an end to end workflow that would
detect genes using test data.

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

### Target generation for training

From GFF and annotations - we are marking 500 bps upstream of the START codon
(first codon in the first CDS of a gene) as 5' UTR, and 500 bps downstream of
the STOP codon (last codon in the last CDS of a gene) as 3' UTR. We are
essentially ignoring the gene annotations in the GFF, and just inferring UTRs,
START, and STOP from the CDS/exons annotations. This is because gene
annotations in GFFs may contain UTRs or may not and is not consistent.

Let's not worry about overlapping genes. In the future, we can look at
predictions with less scores and use them to put together overlapping genes
(and alternative splicing).

We'd like this model to predict 5' UTRs, START, STOP, and 3' UTRs. Each
position of the genomic sequence should be categorized into those 4 categories
plus CDS region or intergenic region.

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

### Current performance and what we are trying
     
Options:
   --config capped (default): Use capped class weights (50x max ratio)
   --config focal: Use focal loss for class imbalance
   --config original: Use original extreme weights (for comparison)

Current model performance

  * From 100% intergenic predictions to 49.9% gene body detection
  * Model can now distinguish between different genomic regions
  * START/STOP codon detection is working (though still needs improvement)
  * UTR regions are being recognized
  * The model is now actually learning gene boundaries rather than just
    predicting everything as intergenic space. This is a foundational
    improvement that validates our approach and sets the stage for further
    refinement through:
      * Additional training epochs
      * Fine-tuning class weights for START/STOP codons
      * More sophisticated loss functions
      * Larger training datasets


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


## To Cursor: Development Rules to Follow

Always use chop_env virtual env.

Don't implement anything without a discussion. First question should be "would
you like me to propose a solution", rather than "would you like me to
implement". What would you implement w/o an agreed upon approach?

Don't leave comments that only pertain to an action you took, e.g. to undo
something. Comments should describe the current state of the code or
configuration.

Because a lot of steps may be compute intensive we want to reuse results, so
don't use temp directories that will get cleaned up automatically. Put them in
directories we can access later.

Run unit tests using

```
cd /Users/benjie/git/chop && scripts/run-unit-tests
```

Whenever we are done implement a set of logic and validated that they are what
we want, we should add unit tests and update the tests/unit/run_tests.py script
to include the new unit tests.

When I ask you a question on why something is the way it is, do not assume I
want you to change it. Tell me if you want to change something and ask if that
is a change I want.

Remember the purpose of a set of changes - and if that set of changes happen to
break something, the most important thing is NOT to fix what broke, but to fix
what broke AND preserve the purpose of the changes.

VERY IMPORTANT - don't change the configuration files and assume the code
supports the syntax changes in the configuration YAML files. Always check to
make sure if you add/modify the structure of the configuration YAML files, we
have code that supports the changes.
