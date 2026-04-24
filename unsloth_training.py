"""
Loan Underwriting AI - Autonomous Fine-Tuning Script
Using Unsloth and TRL for memory-efficient training on Llama-3-8B.

COLAB SETUP:
Run this block first to install dependencies:
!pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
"""

import os
import json
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer

# --- 1. CONFIGURATION ---
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 512
DTYPE = None 
LOAD_IN_4BIT = True

# --- 2. DATASET GENERATION ---
def get_training_data():
    """Generates synthetic banking cases for training."""
    data = [
        {
            "profile": "Name: Rajesh Kumar, Income: ₹1,200,000, Credit Score: 785, Debt: ₹150,000, Loan: ₹500,000, Employment: salaried",
            "response": {
                "risk_level": "Low",
                "loan_decision": "Approve",
                "reasoning": "High credit score and low DTI ratio (12.5%) indicate excellent repayment capacity."
            }
        },
        {
            "profile": "Name: Priya Singh, Income: ₹720,000, Credit Score: 665, Debt: ₹280,000, Loan: ₹400,000, Employment: self_employed",
            "response": {
                "risk_level": "Medium",
                "loan_decision": "Conditional Approve",
                "reasoning": "Moderate credit score. DTI is acceptable (38%), but self-employment requires document verification."
            }
        },
        {
            "profile": "Name: Arjun Das, Income: ₹420,000, Credit Score: 572, Debt: ₹400,000, Loan: ₹300,000, Employment: salaried",
            "response": {
                "risk_level": "High",
                "loan_decision": "Reject",
                "reasoning": "Low credit score and extremely high DTI ratio (>90%). High risk of default."
            }
        }
    ]
    
    formatted = []
    for item in data:
        text = f"### Instruction:\nEvaluate this loan application: {item['profile']}\n\n### Response:\n{json.dumps(item['response'])}"
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
def add_eos(examples):
    return {"text": [t + tokenizer.eos_token for t in examples["text"]]}
dataset = dataset.map(add_eos, batched=True)

# --- 5. TRAINING ---
print("⚙️ Setting up Trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
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
test_profile = "Name: Rahul Verma, Income: ₹600,000, Credit Score: 620, Debt: ₹300,000, Loan: ₹200,000"
inputs = tokenizer([f"### Instruction:\nEvaluate this loan application: {test_profile}\n\n### Response:\n"], return_tensors = "pt").to("cuda")

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
