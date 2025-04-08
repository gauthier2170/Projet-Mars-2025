# 💡 Projet GPT-2 - Fine-tuning en français

Ce projet montre comment fine-tuner GPT-2 sur des textes en français, à l'aide de Hugging Face.

---

## 🔍 Partie 1 : Chargement et préparation des données

On utilise ici le dataset `rayml/french_gutenberg`, qui contient des livres traduits en français issus du projet Gutenberg.

---

### 🧪 Installation des bibliothèques nécessaires

```bash
!pip install transformers torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

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
    print("\n", instruction)
    print(output)
```
# 💡 Projet GPT-2 - Fine-tuning en français

Ce projet montre comment fine-tuner GPT-2 sur des textes en français, à l'aide de Hugging Face.

---

## 📄 Télécharger le code en PDF

👉 [Clique ici pour ouvrir le PDF contenant le code](1ER%20partie%20gpt-2.pdf)

---

## 🔍 Partie 1 : Chargement et préparation des données

### 🧪 Installation

```bash
!pip install torch datasets transformers tqdm matplotlib

