Premier League Match Report Generator
Enoch Obikoya

I put this project together to turn raw Premier League match scores and goal lists into clean, one sentence match summaries. A big reason I built this was because I wanted to use LoRA and learn how it works hands-on for adapting language models to a specific task.

Smaller open source models often mess up when you hand them match stats. When I tested the base model on simple scores, it would constantly hallucinate details, mix up which team actually won, or make up competitions that never happened. To fix this, I fine tuned a small 500M parameter model (Qwen 2.5) using LoRA so that it sticks strictly to the actual scoreline and teams provided.

How It Works

The workflow is broken down into a few practical steps:

1. Data preparation: A script generates synthetic fixtures with realistic scores, goalscorers, and minute marks, paired with verified match summaries.
2. Fine tuning: I used LoRA to train adapter weights on the attention layers while keeping the rest of the base model frozen. This keeps memory usage low enough to train directly on a standard CPU.
3. Verification: A comparison script runs unseen fixtures through both the base model and my fine tuned adapter side by side so you can clearly see the improvement.
4. Testing: An interactive script where you can type or paste any match result and test the model immediately.

Project Files

fetch_and_prepare_dataset.py generates the training and test files.

train.py runs the LoRA training and saves the adapter weights into the football_model_lora folder.

compare.py compares outputs from the baseline model against the fine tuned model on test matches.

test_prompt.py lets you test custom match inputs directly in your terminal or Spyder console.

pipeline_interactive.py runs all three main steps in order if you want to execute everything in one go.

train.jsonl contains the match pairs used for training.

test.jsonl contains the unseen matches used for testing.

requirements.txt lists the Python packages needed.

Setup

Install the required libraries with pip:

pip install -r requirements.txt

Running the Project

Run each step in order from your terminal or IDE:

Step 1: Create the dataset
python fetch_and_prepare_dataset.py

Step 2: Train the model
python train.py

Step 3: Run the comparison
python compare.py

Step 4: Test prompts
python test_prompt.py

Results

The difference between the base model and the fine tuned version is noticeable right away. For example, on a 2-2 draw between West Ham and Manchester City, the base model claimed Manchester City won and that Paqueta scored twice for them. After training, the model correctly summarized that the teams shared the points in an entertaining 2-2 draw. It reliably picks up on clean sheets, draws, and close wins without making up false events.
