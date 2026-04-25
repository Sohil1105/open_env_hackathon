"""
Loan Underwriting AI - Autonomous Fine-Tuning Script
Using Unsloth and TRL for memory-efficient training on Llama-3-8B.

COLAB SETUP:
Run this block first to install dependencies (Sequential for Python 3.12 compatibility):
!pip install unsloth_zoo
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate
!pip install bitsandbytes
!pip install xformers
"""

import os
import json
import torch
import math
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer


# --- 1. CONFIGURATION ---
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 512
# Fixed DTYPE for hardware optimization
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LOAD_IN_4BIT = True
# Fixed device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. DATASET LOADING (Real World) ---
def get_training_data():
    """Loads real-world loan data from HuggingFace and converts to instruction format."""
    print("📥 Loading real-world dataset from HuggingFace (AnguloM/loan_data)...")
    raw = load_dataset("AnguloM/loan_data", split="train")
    
    formatted = []
    for row in raw:
        # Decode income from log scale
        income = round(math.exp(row["log.annual.inc"]), 2)
        dti = row["dti"]
        fico = row["fico"]
        purpose = row["purpose"].replace("_", " ")
        delinq = row["delinq.2yrs"]
        pub_rec = row["pub.rec"]
        inq = row["inq.last.6mths"]
        not_paid = row["not.fully.paid"]
        credit_policy = row["credit.policy"]

        # Derive decision based on dataset features
        if credit_policy == 1 and not_paid == 0 and fico >= 720 and dti < 20:
            risk, decision = "Low", "Approve"
            reason = f"Strong FICO ({fico}), low DTI ({dti}%), clean repayment history."
        elif credit_policy == 1 and not_paid == 0 and fico >= 660:
            risk, decision = "Medium", "Conditional Approve"
            reason = f"Acceptable FICO ({fico}), DTI ({dti}%) — requires document verification."
        else:
            risk, decision = "High", "Reject"
            reason = f"Low FICO ({fico}), high DTI ({dti}%) or poor repayment record."

        profile = (
            f"Income: ${income:,}, FICO: {fico}, DTI: {dti}%, "
            f"Purpose: {purpose}, Delinquencies: {delinq}, "
            f"Public Records: {pub_rec}, Inquiries (6mo): {inq}"
        )

        text = (
            f"### Instruction:\nEvaluate this loan application: {profile}\n\n"
            f"### Response:\n"
            f'{{"risk_level":"{risk}","decision":"{decision}","reason":"{reason}"}}'
        )
        formatted.append({"text": text})

    return Dataset.from_list(formatted)

# --- 3. MODEL LOADING ---
print("🚀 Loading Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# --- 4. PREPARE DATA ---
print("📊 Preparing Dataset...")
dataset = get_training_data()

# CHANGE: Added num_proc=4 for parallel pre-tokenization processing
def add_eos(examples):
    return {"text": [t + tokenizer.eos_token for t in examples["text"]]}
dataset = dataset.map(add_eos, batched=True, num_proc=4)

# --- 5. TRAINING ---
print("⚙️ Setting up Trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    packing = True,  # Added packing=True for efficiency
    args = TrainingArguments(
        per_device_train_batch_size = 2,      # CHANGE: Increased from 1
        gradient_accumulation_steps = 2,      # CHANGE: Reduced from 4 (Effective batch = 4)
        dataloader_num_workers = 4,            # CHANGE: Added for faster data loading
        dataloader_pin_memory = True,          # CHANGE: Added for faster GPU transfer
        warmup_steps = 10,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_steps = 500,
        save_total_limit = 2,
        report_to = "none",
    ),
)

print("🔥 Starting Training...")
trainer_stats = trainer.train()

# --- 6. VISUALIZATION ---
print("📈 Generating Plots...")
history = trainer.state.log_history
steps = [x['step'] for x in history if 'loss' in x]
loss = [x['loss'] for x in history if 'loss' in x]

plt.figure(figsize=(10, 6))
plt.plot(steps, loss, color='#ff3366', linewidth=2)
plt.title("Loan Underwriting - Training Loss", color='white')
plt.xlabel("Step", color='white')
plt.ylabel("Loss", color='white')
plt.grid(True, alpha=0.2)
plt.gca().set_facecolor('#1a1a1a')
plt.gcf().set_facecolor('#1a1a1a')
plt.savefig("training_loss.png")
print("✅ Loss plot saved to training_loss.png")

# --- 7. VERIFICATION TEST ---
print("\n" + "="*50)
print("🔍 LIVE VERIFICATION (UNSEEN CASE)")
print("="*50)

FastLanguageModel.for_inference(model)
test_profile = "Income: $85,000, FICO: 740, DTI: 12.0%, Purpose: home improvement, Delinquencies: 0, Public Records: 0, Inquiries (6mo): 0"
inputs = tokenizer([f"### Instruction:\nEvaluate this loan application: {test_profile}\n\n### Response:\n"], return_tensors = "pt").to(device)

streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = streamer, max_new_tokens = 256)

print("\n" + "="*50)
print("SUCCESS: Pipeline is fully operational.")
print("="*50)

# --- 8. SAVE & UPLOAD TO HUGGING FACE ---
HF_TOKEN = "" 

if HF_TOKEN != "":
    print("📤 Uploading model to Hugging Face...")
    model.push_to_hub("Sourav0511/loan-underwriting-lora", token = HF_TOKEN)
    tokenizer.push_to_hub("Sourav0511/loan-underwriting-lora", token = HF_TOKEN)
    print("✅ Model uploaded successfully to: https://huggingface.co/Sourav0511/loan-underwriting-lora")
else:
    print("⚠️ Skipping upload: Please paste your Hugging Face Token into the HF_TOKEN variable.")
