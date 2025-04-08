# Projet-Mars-2025
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # nécessaire pour éviter les erreurs de padding

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

examples = [
    "What is the capital of France?",
    "What are the three primary colors?",
    "What does DNA stand for?"
]

for instruction in examples:
    prompt = f"Instruction: {instruction}\nRéponse:"
    output = generator(prompt, max_new_tokens=60)[0]["generated_text"]
    print("\n➡️", instruction)
    print(output)
