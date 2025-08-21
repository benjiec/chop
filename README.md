# CHOP: Chromosome Open Prediction

A transformer-based gene prediction tool that uses attention mechanisms to identify genes in genomic DNA sequences.

## Python Virtual Env

Make a Python env

```
# Create venv
python3 -m venv chop_env
source chop_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If you are on M2

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Train with M2 optimization - note the yaml file contain the data file paths

```
python training/train.py --config configs/m2_native.yaml
```

