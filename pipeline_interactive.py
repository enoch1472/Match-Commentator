# -*- coding: utf-8 -*-
"""
Football Match Report Generator - Interactive Full Pipeline
Executes dataset generation, LoRA fine-tuning, and model comparison in sequence.
"""

from fetch_and_prepare_dataset import build_dataset
from train import train_model
from compare import run_comparison

print("Step 1: Generating dataset...")
build_dataset()

print("\nStep 2: Training LLM with LoRA...")
train_model()

print("\nStep 3: Comparing before vs after...")
run_comparison()
