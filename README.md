# 💡 Projet GPT-2 - Fine-tuning en français

Ce projet montre comment fine-tuner GPT-2 sur des textes en français, à l'aide de Hugging Face.

---

## 🔍 Partie 1 : Chargement et préparation des données

On utilise ici le dataset `rayml/french_gutenberg`, qui contient des livres traduits en français issus du projet Gutenberg.

---

### 🧪 Installation des bibliothèques nécessaires

```bash
pip install torch datasets transformers tqdm matplotlib
