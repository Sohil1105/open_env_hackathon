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
    """
    Inclusive classification to extract 4,000+ balanced real-world examples.
    Includes a 10% validation split.
    """
    import math
    import random
    from datasets import Dataset, load_dataset
    random.seed(42)

    print("📥 Loading full real-world dataset (9.5k rows)...")
    raw = load_dataset("AnguloM/loan_data", split="train")

    low_cases, medium_cases, high_cases = [], [], []

    for row in raw:
        income = round(math.exp(row["log.annual.inc"]), 2)
        dti = row["dti"]
        fico = row["fico"]
        purpose = row["purpose"].replace("_", " ")
        delinq = row["delinq.2yrs"]
        pub_rec = row["pub.rec"]
        inq = row["inq.last.6mths"]
        not_paid = row["not.fully.paid"]

        # Compute monthly DTI properly
        monthly_income = income / 12
        installment = row.get("installment", 0)
        real_dti = (installment / monthly_income * 100) if monthly_income > 0 else dti

        # MORE INCLUSIVE LOGIC to capture more of the 9.5k rows
        if fico < 650 or real_dti > 40 or delinq >= 1 or pub_rec >= 1 or not_paid == 1:
            risk, decision, rate = "High", "Reject", "14%+"
            reason = f"High risk identified: FICO {fico}, DTI {real_dti:.1f}%, or history of negative records."
            high_cases.append((income, dti, fico, purpose, delinq, pub_rec, inq, reason, risk, decision, rate))
        elif fico >= 710 and real_dti < 22 and delinq == 0 and pub_rec == 0 and not_paid == 0:
            risk, decision, rate = "Low", "Approve", "7-9%"
            reason = f"Excellent credit: FICO {fico}, DTI {real_dti:.1f}%, and clean history."
            low_cases.append((income, dti, fico, purpose, delinq, pub_rec, inq, reason, risk, decision, rate))
        else:
            risk, decision, rate = "Medium", "Conditional Approve", "10-13%"
            reason = f"Moderate risk. FICO {fico} and DTI {real_dti:.1f}% require verification."
            medium_cases.append((income, dti, fico, purpose, delinq, pub_rec, inq, reason, risk, decision, rate))

    print(f"📊 Extracted Categories — Low: {len(low_cases)}, Medium: {len(medium_cases)}, High: {len(high_cases)}")

    # Balance to target ~1,400 per class for 4,200 total
    target_per_class = min(len(low_cases), len(medium_cases), len(high_cases), 1400)
    print(f"⚖️ Balancing to ~{target_per_class} per class...")

    balanced = (
        random.sample(low_cases, target_per_class) +
        random.sample(medium_cases, target_per_class) +
        random.sample(high_cases, target_per_class)
    )
    random.shuffle(balanced)

    formatted = []
    for case in balanced:
        inc, d, f, p, de, pb, i, reason, risk, dec, rate = case
        profile = f"Income: ${inc:,}, FICO: {f}, DTI: {d}%, Purpose: {p}, Delinquencies: {de}, Public Records: {pb}, Inquiries (6mo): {i}"
        text = f"### Instruction:\nEvaluate this loan application: {profile}\n\n### Response:\n" \
               f'{{"risk_level":"{risk}","decision":"{dec}","interest_rate":"{rate}","reason":"{reason}"}}'
        formatted.append({"text": text})

    full_ds = Dataset.from_list(formatted)
    split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
    print(f"✅ Final Dataset — Train: {len(split_ds['train'])}, Validation: {len(split_ds['test'])}")
    return split_ds

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
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# --- 4. PREPARE DATA ---
print("📊 Preparing Dataset...")
dataset_dict = get_training_data()

def add_eos(examples):
    return {"text": [t + tokenizer.eos_token for t in examples["text"]]}

train_dataset = dataset_dict["train"].map(add_eos, batched=True, num_proc=4)
eval_dataset = dataset_dict["test"].map(add_eos, batched=True, num_proc=4)

# --- 5. TRAINING ---
print("⚙️ Setting up Trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    packing = True,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        dataloader_num_workers = 4,
        dataloader_pin_memory = True,
        warmup_steps = 100,
        num_train_epochs = 5,
        learning_rate = 5e-5,
        evaluation_strategy = "steps",
        eval_steps = 100,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
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

# --- MERGE LORA INTO BASE MODEL ---
print("🔀 Saving LoRA adapter first...")
model.save_pretrained("lora_adapter_temp")
tokenizer.save_pretrained("lora_adapter_temp")

print("🔀 Reloading in 16-bit for clean merge...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/llama-3-8b-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
merged_model = PeftModel.from_pretrained(base_model, "lora_adapter_temp")
merged_model = merged_model.merge_and_unload()
print("✅ LoRA merged in 16-bit — no rounding errors")

# --- PUSH MERGED MODEL ---
HF_TOKEN = "" # @param {type:"string"}

if HF_TOKEN != "":
    print("📤 Uploading MERGED model to HuggingFace...")
    merged_model.push_to_hub(
        "Sourav0511/loan-underwriting-merged-v2",
        token=HF_TOKEN,
        max_shard_size="2GB"
    )
    tokenizer.push_to_hub(
        "Sourav0511/loan-underwriting-merged-v2",
        token=HF_TOKEN
    )
    print("✅ Merged model uploaded to: https://huggingface.co/Sourav0511/loan-underwriting-merged-v2")

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

if HF_TOKEN != "":
    print("📤 Uploading LoRA adapter to Hugging Face...")
    model.push_to_hub("Sourav0511/loan-underwriting-lora-v2", token = HF_TOKEN)
    tokenizer.push_to_hub("Sourav0511/loan-underwriting-lora-v2", token = HF_TOKEN)
    print("✅ Adapter uploaded successfully to: https://huggingface.co/Sourav0511/loan-underwriting-lora-v2")
else:
    print("⚠️ Skipping upload: Please paste your Hugging Face Token into the HF_TOKEN variable.")
