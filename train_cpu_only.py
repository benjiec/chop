#!/usr/bin/env python3
"""
Training script that forces CPU-only usage to avoid MPS issues.
"""

import os
import torch

# Force CPU usage - disable MPS completely
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Explicitly disable MPS
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

# Force CPU as default device
torch.set_default_device('cpu')

# Now import and run the normal training
import sys
sys.path.append('.')

from training.train import main

if __name__ == '__main__':
    # Add CPU-only argument to sys.argv if not present
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', 'configs/m2_cpu_only.yaml'])
    
    print("🚀 Starting CPU-only training (MPS disabled)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'Not supported'}")
    print(f"Default device: {torch.get_default_device() if hasattr(torch, 'get_default_device') else 'cpu'}")
    
    main()
