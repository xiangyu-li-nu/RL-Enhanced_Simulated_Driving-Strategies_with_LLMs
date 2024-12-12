import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
with open(r"D:\LLM_for_AV\Code\filtered_train_new.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert the list to a HuggingFace Dataset
dataset = Dataset.from_list(data)

# Load the PEFT configuration and the trained model
peft_model_id = "/root/autodl-tmp/peft_model_8b"  # Replace this with the path to your trained model
config = PeftConfig.from_pretrained(peft_model_id)

# Load the base model (e.g., T5, BART, or other Seq2Seq models)
model_name = "/root/autodl-tmp/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the PEFT model (i.e., the trained LoRA model)
model = PeftModel.from_pretrained(model, peft_model_id)

# Load the tokenizer (ensure it is compatible with the model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set the pad_token for the tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as the padding value

# # Load the tokenizer and model
# model_name = r"D:\LLM_for_AV\llama3.1-7b"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Set pad_token for the tokenizer
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as padding token
#
# # Load the pretrained model onto GPU
# model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # LoRA dropout rate
    task_type="CAUSAL_LM",  # Task type
)

# Apply the LoRA configuration
model = get_peft_model(model, lora_config)

# Move the model to GPU
model = model.to(device)

# # Disable caching to support gradient checkpointing
# model.config.use_cache = False
#
# # Enable gradient checkpointing to reduce memory usage
# model.gradient_checkpointing_enable()

model.print_trainable_parameters()

# Define the data preprocessing function
def tokenize_function(examples):
    input_texts = examples['question']
    target_texts = examples['answer']
    model_inputs = tokenizer(
        input_texts, padding='max_length', truncation=True, max_length=512
    )
    labels = tokenizer(
        target_texts, padding='max_length', truncation=True, max_length=512
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and validation sets
dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']

# Set training parameters
num_epochs = 5
optimizer = AdamW(model.parameters(), lr=5e-5)

# Start training
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataset:
        # Move the input data and labels to GPU
        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0).to(device)  # Add batch dimension
        labels = torch.tensor(batch['labels']).unsqueeze(0).to(device)  # Add batch dimension

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss  # Compute the loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs} completed. Total Loss: {total_loss}")

# Save the fine-tuned model
model.save_pretrained('./peft_model')
tokenizer.save_pretrained('./peft_model')
