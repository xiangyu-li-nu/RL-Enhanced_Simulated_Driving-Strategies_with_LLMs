from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Load the PEFT configuration and the trained model
peft_model_id = "peft_model"  # Replace this with the path to your trained model
config = PeftConfig.from_pretrained(peft_model_id)

# Load the base model (e.g., T5, BART, or other Seq2Seq models)
model_name = "D:/Pretrained_Models/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the PEFT model (i.e., the trained LoRA model)
model = PeftModel.from_pretrained(model, peft_model_id)

# Load the Tokenizer (ensure it is compatible with the model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token for tokenizer if it is not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as padding token

# Inference
model.eval()  # Set the model to evaluation mode
input_text = "This is a test sentence."  # Replace with your input text

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Perform inference with the model
with torch.no_grad():  # Disable gradient calculation to save memory
    generated_ids = model.generate(inputs["input_ids"])

# Convert the generated token ids back to text
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated Text: {generated_text}")
