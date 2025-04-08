# 💡 Projet GPT-2 - Fine-tuning 

Ce projet montre comment fine-tuner GPT-2 sur des textes en français, à l'aide de Hugging Face.

---

## 📄 Télécharger le code en PDF

👉 [Clique ici pour ouvrir le PDF contenant le code](1ER%20partie%20gpt-2.pdf)

---

## 🔍 Partie 1 : Chargement et préparation des données

### 🧪 Installation

```bash
!pip install torch datasets transformers tqdm matplotlib
SEQUENCE_LENGTH = 128
BATCH_SIZE = 32

from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch

