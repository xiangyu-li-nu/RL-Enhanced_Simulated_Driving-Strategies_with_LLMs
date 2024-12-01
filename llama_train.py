import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# Load Dataset
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert a List to a HuggingFace Dataset
dataset = Dataset.from_list(data)

# Loading the tokenizer and model
model_name = "D:\LLM_for_AV\Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token for tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as fill value

# Loading pre-trained models to the CPU
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:1")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

# Disable caching to support gradient checkpointing
model.config.use_cache = False

# Enable gradient checkpointing to reduce memory overhead
model.gradient_checkpointing_enable()

# Configuring LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,  # Reduce r to reduce memory requirements
    lora_alpha=16,
    lora_dropout=0.2
)

# Applying LoRA to the model
model = get_peft_model(model, lora_config)

# Define data preprocessing functions
def tokenize_function(examples):
    input_texts = examples['question']
    target_texts = examples['answer']
    model_inputs = tokenizer(
        input_texts, padding='max_length', truncation=True, max_length=1024
    )
    labels = tokenizer(
        target_texts, padding='max_length', truncation=True, max_length=1024
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocessing the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Setting training parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=30,
    per_device_train_batch_size=1,  # Reduce batch size
    gradient_accumulation_steps=4,  # Gradient accumulation to simulate larger batches
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    eval_strategy="no",
    learning_rate=5e-5,
    fp16=False,  # Disable mixed precision (not supported on CPU)
    dataloader_num_workers=0,  # On Windows, this should be set to 0.
    no_cuda=True  # Force CPU usage
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model('./fine-tuned-model')

