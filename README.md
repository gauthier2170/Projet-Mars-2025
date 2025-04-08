# 💡 Projet GPT-2 - Fine-tuning en français

Ce projet montre comment fine-tuner GPT-2 sur des textes en français, à l'aide de Hugging Face.

---

## 🔍 Partie 1 : Chargement et préparation des données

On utilise ici le dataset `rayml/french_gutenberg`, qui contient des livres traduits en français issus du projet Gutenberg.

---

### 🧪 Installation des bibliothèques nécessaires

```bash
pip install torch datasets transformers tqdm matplotlib
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(element):
    return tokenizer(element["text"])

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_len = (len(concatenated) // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
    result = [
        concatenated[i : i + SEQUENCE_LENGTH]
        for i in range(0, total_len, SEQUENCE_LENGTH)
    ]
    return {"input_ids": result, "labels": result}

dataset = dataset.map(group_texts, batched=True)
