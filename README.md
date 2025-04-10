# Projet : Aligner GPT-2 pour en faire un assistant (Instruction Tuning)

L'objectif de ce projet est de transformer un modèle de langage tel que GPT-2 en un assistant qui sera capable de répondre à des questions. 
Pour cela, on va utiliser GPT-2 et nous allons le fine-tuner avec un dataset (Alpaca) et un apprentissage efficace via LoRA (Low-Rank Adaptation).
Tout d'abord, nous chargerons et nous testerons le modèle gpt2-medium. Puis, nous téléchargerons un dataset et nous le tokeniserons. 
Ensuite, nous ferons un fine-tuning avec LoRa. Enfin, nous comparerons les réponses des modèles avant et après le fine-tuning. 

---

## Partie 1 : Chargement gpt2-medium et test

Tout d'abord, nous voulons tester un modèle de GPT-2, gpt2-medium. Nous allons générer des questions ou des instructions précisent avec des prompts et nous voudrions savoir si ce modèle y répond correctement. 
Nous allons constater que ce n'est pas le cas et que cela répond soit des mots qui ne forment même pas une phrase ou bien cela répète la question posée. 
Le code permet de charger gpt-2 medium et son tokenizer. Il y a aussi la création de pipeline afin de générer du texte automatiquement à partir des prompts.

---

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
    "Traduit : What happened ?",
    "Give three tips for staying healthy."    
]

for instruction in examples:
    prompt = f"Instruction: {instruction}\nRéponse:"
    output = generator(prompt, max_new_tokens=60)[0]["generated_text"]
    print("\n", instruction)
    print(output)
```

## Partie 2 : Charger le dataset et le tokenizer
### 2.1. Dataset

Maintenant, nous allons télécharger le dataset de Alpaca disponible sur Hugging Face Datasets. C'est un dataset avec des questions simples et des réponses assez courtes. Nous utiliserons seulement les colonnes instruction et output, qui correspond aux réponsent des instructions. Nous nous intéressons seulement à ces deux colonnes car les autres ne sont pas utiles pour notre projet. De plus, nous réduisons les réponses à 80 mots maximums pour éviter au modèle de se perdre dans des réponses trop longues et cela permet d'accélérer l'entrainement. Nous voulons poser des questions et avoir des réponses qui ressemble à celles du dataset. 

---

```bash
!pip install datasets

from datasets import load_dataset

# Charger Alpaca, en gardant seulement 'instruction' et 'output'
dataset = load_dataset("tatsu-lab/alpaca")["train"]
dataset = dataset.remove_columns(["input"])

# Optionnel : filtrer les réponses longues (> 80 mots)
def is_simple(example):
    return len(example["output"].split()) <= 80

dataset = dataset.filter(is_simple)
```

### 2.2. Tokenizer

Ce qui est important dans cette étape c'est de bien réaliser la tokenisation, il faut l'adapter à notre dataset et comment on veut l'utiliser. C'est pour cela que l'on créer un prompt au format Alpaca avec instruction et réponse. Puis, nous le passons dans le tokenizer avec une limite de 128 tokens car les questions et réponses sont courtes et cela permet d'acccelérer l'entrainement. Enfin, on applique la tokenization sur le dataset et on obtient le texte tokenizé appelé dans le code tokenized_dataset. 

---

```bash
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 n'a pas de token de padding à la base

def tokenize(example):
    prompt = f"Instruction: {example['instruction']}\nRéponse:"
    full_text = prompt + " " + example["output"]
    tokenized = tokenizer(full_text, truncation=True, max_length=128, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
base_model.resize_token_embeddings(len(tokenizer))  # Pour intégrer le token de padding
```
# Partie Quentin sur la config de LoRa

## Partie 3 : Fine-tuning avec LoRa
### 3.1. Configuration de LoRa

---
```bash
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
```
# Partie Quentin sur le trainer
### 3.2. L'entraimenent

---
```bash
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

output_dir = "./gpt2-medium-alpaca-lora"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=1000,
    learning_rate=3e-4,
    num_train_epochs=2,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    save_total_limit=1,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

## Partie 4 : Comparaison des réponses
### 4.1. Premier prompt de vérification

Dans un premier temps, nous avons réalisé un premier prompt avec une seule question afin de vérifier que le fine-tuning et la sauvegarde avaient bien fonctionné. Nous avons avons posé une question par rapport au modèle sur lequel on l'a entrainé. De plus, on utilise de nouveau un pipeline de génération de texte de Hugging Face. Enfin, quand on lance la génération de la réponse, on peut contrôler plusieurs choses, les plus importantes sont par exemple le nombre de tokens maximals pour éviter les réponses trop longue (ici 80 tokens), la température permet de gérer la créativité des réponses. 

---

```bash
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

instruction = "What is the capital of France?"
prompt = f"Instruction: {instruction}\nRéponse:"
result = pipe(
    prompt,
    max_new_tokens=80,
    temperature=0.7,      # plus conservateur
    top_p=0.9,            # nucleus sampling
    top_k=50,             # coupe les options absurdes
    eos_token_id=tokenizer.eos_token_id
)[0]["generated_text"]

print("\n Réponse générée :\n")
print(result)
```

### 4.2. Comparaison avant/après fine-tuning

Maintenant, nous allons comparer les réponses générées par deux modèles : gpt-2 medium et gpt-2 medium fine-tuné avec LoRa sur le dataset Alpaca. Nous cherchons à savoir si le fine-tuning a été de qualité ou non sur les questions que l'on va poser du dataset Alpaca. Tout d'abord, nous avons posé plusieurs questions provenant du datset avec des instructions différentes. Ensuite, on a encore la partie avec les paramètres qu'il faut bien réglés pour avoir un meilleur contrôle. Puis, on charge tous nos modèles qu'on veut tester ici, c'est à dire le modèle fine-tuné et celui qui ne l'est pas. Enfin, on génère les réponses avec l'outil pipeline de Hugging Face puis on peut lancer les comparaisons. Nous avons posé la question et mis la réponse de gpt-2 medium puis celle avec le fine-tuning pour observer les différences. On a pu constater que le modèle de base répond de manière très aléatoire, les mots s'enchainent mais la phrase ne veut rien dire et parfois il répète plusieurs fois les mêmes mots à la suite. Par contre, le modèle fine-tuné est cohérent, il commence par nous répondre à notre question et donne d'autres informations en plus qui sont pertinentes par rapport à la question. Cependant, ses réponses sont parfois très longues et perdent de la clartée au bout d'un moment, c'est pour cela que nous avons réduit le nombre de mots maximums des réponses. On en conclu que le fine-tuning est fonctionnel puisqu'il est pertinent, claire et utile dans beaucoup de situation de questions/réponses style Alpaca. 

```bash
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Instructions de test
eval_instructions = [
    "What is the capital of France?",
    "Give three tips for staying healthy.",
    "Create a list of five different animals",
    "Who wrote 'Romeo and Juliet'?"
]

# Paramètres de génération (réglés pour du bon contrôle)
gen_kwargs = dict(
    max_new_tokens=80,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=50256  # Fin de séquence pour GPT2
)

# Charger modèle de base
model_base = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer_base = AutoTokenizer.from_pretrained("gpt2-medium")

# Charger modèle fine-tuné
model_ft = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer_ft = AutoTokenizer.from_pretrained("gpt2-medium")
model_ft.resize_token_embeddings(len(tokenizer_ft))
model_ft = PeftModel.from_pretrained(model_ft, "./gpt2-medium-alpaca-lora")

# Pipelines
pipe_base = pipeline("text-generation", model=model_base, tokenizer=tokenizer_base, device_map="auto")
pipe_ft = pipeline("text-generation", model=model_ft, tokenizer=tokenizer_ft, device_map="auto")

# Comparaison
for instr in eval_instructions:
    prompt = f"Instruction: {instr}\nRéponse:"

    base_resp = pipe_base(prompt, **gen_kwargs)[0]["generated_text"]
    ft_resp = pipe_ft(prompt, **gen_kwargs)[0]["generated_text"]

    # Affichage propre
    print("\n" + "="*80)
    print(f" Instruction: {instr}")
    print("\n Réponse GPT2 (base):")
    print(base_resp.replace(prompt, "").strip())

    print("\n Réponse GPT2 fine-tuné:")
    print(ft_resp.replace(prompt, "").strip())
    print("="*80)
```

### 4.3. Interface de questions/réponses 

Cette interface interactive permet de répondre aux questions qu'on lui pose sur le dataset Alpaca en utilisant notre modèle fine-tuné GPT-2 avec LoRa. En effet, vous allez avoir une barre où vous pouvez taper votre question (dataset Alpaca) et obtenir une réponse immédiate à celle-ci. On peut le faire un grand nombre de fois et lorsque vous voulez vous arrêter il suffit de taper exit dans la barre. Cela permet une assistance IA facile d'utilisation et qui obtient de bon résultats de réponse clair et pertinente.  


```bash
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Charger modèle fine-tuné
base_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
base_model.resize_token_embeddings(len(tokenizer))

# Charger les poids LoRA
model = PeftModel.from_pretrained(base_model, "./gpt2-medium-alpaca-lora")

# Pipeline avec réglages pour qualité
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Paramètres de génération
gen_kwargs = dict(
    max_new_tokens=100,
    temperature=0.7,      # Gère la créativité (0.7 = bon équilibre)
    top_p=0.9,            # Nucleus sampling
    do_sample=True,
    eos_token_id=50256    # Pour forcer l’arrêt en fin de phrase
)

# Interface utilisateur
print("Assistant Fine-tuné Alpaca | Tape 'exit' pour quitter\n")

while True:
    instr = input(" Instruction: ")
    if instr.lower() in ["exit", "quit"]:
        break

    prompt = f"Instruction: {instr}\nRéponse:"

    try:
        result = pipe(prompt, **gen_kwargs)[0]["generated_text"]
        print("\n Réponse générée :")
        print(result.replace(prompt, "").strip())
        print("=" * 80 + "\n")
    except Exception as e:
        print(" Erreur :", e)

```
