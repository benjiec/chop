#!/usr/bin/env python3
"""
Training script that enables MPS (Metal Performance Shaders) acceleration for Apple Silicon.
Falls back to CPU if MPS is not available or encounters issues.
"""

import os
import sys
import torch
import warnings

# Enable MPS optimizations - let PyTorch handle memory management automatically

def setup_mps_device():
    """Setup and validate MPS device. Fails if MPS is not available."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("❌ MPS not available on this system. Use train_cpu_only.py instead.")
    
    if not torch.backends.mps.is_built():
        raise RuntimeError("❌ MPS not built in this PyTorch installation. Use train_cpu_only.py instead.")
    
    try:
        # Test MPS with a simple operation
        device = torch.device('mps')
        
        # Set the random number generator to use MPS
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(42)
        
        # Create tensors on MPS device explicitly
        test_tensor = torch.randn(100, 100).to(device)
        test_tensor2 = torch.randn(100, 100).to(device)
        _ = torch.mm(test_tensor, test_tensor2)  # Simple matrix multiplication test
        
        print("🚀 MPS acceleration enabled!")
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS device: {device}")
        return device
    except Exception as e:
        raise RuntimeError(f"❌ MPS test failed: {e}\nUse train_cpu_only.py instead.")

def check_mps_memory():
    """Check MPS memory usage (if available)."""
    try:
        if torch.backends.mps.is_available():
            # Note: MPS doesn't have direct memory reporting like CUDA
            # This is a placeholder for future PyTorch versions
            print("MPS memory monitoring not yet available in PyTorch")
    except Exception:
        pass

# Setup device (MPS only - will fail if not available)
device = setup_mps_device()

# Don't set default device - let the training script handle device placement
# This avoids the random number generator device mismatch issue

# Set random seed for reproducibility
torch.manual_seed(42)
if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
    torch.mps.manual_seed(42)

print(f"✅ MPS device ready: {device}")

# Import training after device setup
sys.path.append('.')
from training.train import main

if __name__ == '__main__':
    # Add default config if not specified
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', 'configs/m2_mps_model7M_layer4_head6_seqlen12k_sliding.yaml'])
    
    print(f"🎯 Starting MPS-accelerated training")
    print(f"Default device: {device}")
    print("💡 MPS Tips:")
    print("  - Monitor Activity Monitor for GPU usage")
    print("  - If you see memory errors, reduce batch_size")
    print("  - MPS memory is shared with system graphics")
    print("  - For CPU training, use train_cpu_only.py instead")
    
    # Check memory before starting
    check_mps_memory()
    
    try:
        main()
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            print(f"\n❌ MPS error encountered: {e}")
            print("💡 Suggestions:")
            print("  - Try reducing batch_size in your config")
            print("  - Use train_cpu_only.py for CPU training")
            print("  - Check Activity Monitor for memory usage")
            sys.exit(1)
        else:
            raise e
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(0)
