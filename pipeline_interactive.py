from fetch_and_prepare_dataset import build_dataset
from train import train_model
from compare import run_comparison

print("Step 1: Generating Dataset...")
build_dataset()

print("\nStep 2: Training Small LLM with LoRA...")
train_model()

print("\nStep 3: Comparing Before vs After...")
run_comparison()
