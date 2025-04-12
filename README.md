# Projet : Aligner GPT-2 pour en faire un assistant (Instruction Tuning)

L'objectif de ce projet est de transformer un modèle de langage tel que GPT-2 en un assistant qui sera capable de répondre à des questions. Pour cela, nous allons utiliser GPT-2 et puis nous le fine-tunerons avec un dataset (Alpaca) et un apprentissage efficace via LoRA (Low-Rank Adaptation). Tout d'abord, nous chargerons et nous testerons le modèle gpt2-medium. Puis, nous téléchargerons un dataset et nous le tokeniserons. Ensuite, nous ferons un fine-tuning avec LoRA. Enfin, nous comparerons les réponses des modèles avant et après le fine-tuning.

---

## Partie 1 : Chargement gpt2-medium et test

Tout d'abord, nous voulons tester un modèle de GPT-2, gpt2-medium. Nous allons générer des questions ou des instructions précises avec des prompts, et nous voudrions savoir si ce modèle y répond correctement. Nous allons constater que ce n’est pas le cas : soit il génère des mots qui ne forment même pas une phrase, soit il répète simplement la question posée. Le code permet de charger gpt2-medium ainsi que son tokenizer. Il inclut également la création d’un pipeline afin de générer automatiquement du texte à partir des prompts.

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

Maintenant, nous allons télécharger le dataset Alpaca, disponible sur Hugging Face Datasets. C’est un dataset contenant des questions simples et des réponses assez courtes. Nous utiliserons seulement les colonnes instruction et output, cette dernière correspondant aux réponses des instructions. Nous nous concentrons uniquement sur ces deux colonnes, car les autres ne sont pas utiles pour notre projet.
De plus, nous limitons les réponses à 80 mots maximum afin d’éviter que le modèle ne se perde dans des réponses trop longues, et cela permet également d’accélérer l’entraînement. L’objectif est de poser des questions et d’obtenir des réponses similaires à celles présentes dans le dataset.

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

Ce qui est important dans cette étape, c’est de bien réaliser la tokenisation : il faut l’adapter à notre dataset et à la manière dont nous voulons l’utiliser. C’est pour cela que nous créons un prompt au format Alpaca, avec une instruction suivie de sa réponse.
Ensuite, nous le passons dans le tokenizer avec une limite de 128 tokens, car les questions et réponses sont courtes, ce qui permet d’accélérer l’entraînement. Enfin, nous appliquons la tokenisation sur le dataset, et nous obtenons le texte tokenisé, appelé tokenized_dataset dans le code.



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

## Partie 3 : Fine-tuning avec LoRa

### 3.1. Explication générale de cette méthode de fine-tuning


La méthode LoRA est une technique efficace de fine-tuning de modèles de langage, qui consiste à ne pas modifier les poids d’origine du modèle (W matrice de taille dxd), mais à apprendre une correction légère sous forme de deux matrices A et B (respectivement de taille dxr et rxd). Plutôt que de recalculer l’intégralité des paramètres du modèle  souvent très volumineux  LoRA introduit ces deux petites matrices dont le produit  A.B (de taille dxd) constitue une mise à jour ΔW appliquée à la matrice d’origine. Ces matrices sont les seules à être entraînées, ce qui réduit drastiquement le nombre de paramètres à optimiser et accélère le processus d’apprentissage. En effet, d est bien supérieur à r il va donc nous falloir optimiser 2.r.d paramètres au lieu de d.d qui est un nombre bien plus important de paramètres. 

Le processus d’apprentissage consiste alors à ajuster les coefficients de A et B afin de minimiser la fonction de perte, qui mesure l’écart entre les prédictions du modèle et les réponses correctes, contenues dans le dataset d’entraînement. Cette fonction de perte va être calculer avec la méthode de Cross Enthropie. À chaque itération, le modèle prédit une réponse, qu’on compare à la vérité issue du dataset. La différence entre les deux est quantifiée par la fonction de perte, ensuite la descente de gradient ajuste progressivement les coefficients des matrices A et B pour que le modèle s’améliore. Une fois entraînées, ces matrices contiennent les coefficients optimaux qui permettent de minimiser notre fonction de perte et permettent au modèle de générer des prédictions précises pour la tâche demandée.

---

### 3.2. Configuration de LoRa

Dans cette partie, nous allons configurer le fine-tuning de notre modèle GPT avec la méthode LoRA (Low-Rank Adaptation) à l’aide de la bibliothèque peft. Pour cela, on définit une configuration via la classe LoraConfig, qui permet de spécifier précisément comment et où appliquer LoRA. On y retrouve plusieurs paramètres clés : le rang r (ici fixé à 8), qui détermine la taille des matrices de décomposition basse-rang A et B; un rang plus petit signifie moins de paramètres à entraîner, mais aussi une capacité d’adaptation plus limitée. Le paramètre lora_alpha (valeur ici de 16) joue le rôle de facteur de mise à l’échelle, amplifiant ou atténuant l’effet de la mise à jour A.B. Le dropout (fixé à 0.1) permet de régulariser l’apprentissage pour éviter l’overfitting. Le champ target_modules précise les sous-couches du modèle ciblées par LoRA, comme ici "c_attn" dans les blocs d’attention de GPT. On indique également si l’on veut entraîner les biais (bias="none") et le type de tâche (task_type="CAUSAL_LM") pour guider l’adaptation. Une fois cette configuration définie, le modèle de base (base_model) est transformé via get_peft_model() pour appliquer LoRA de manière ciblée.

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

### 3.3. L'entraimenent

A présent, nous allons expliquer le code de l'entrainement du modèle via classe Trainer de la bibliothèque Hugging Face. Les paramètres d’entraînement sont définis via TrainingArguments, qui permet de spécifier notamment la taille des batches par appareil (per_device_train_batch_size), le nombre d’époques (num_train_epochs, c’est-à-dire le nombre de fois où le dataset sera parcouru), ou encore le taux d’apprentissage (learning_rate), qui contrôle la vitesse d’ajustement des poids lors de la descente de gradient. Des paramètres comme logging_dir et logging_steps permettent de gérer la journalisation de l’entraînement (en sauvegardant régulièrement les métriques de performance). Pour adapter l’entraînement au language modeling causal (comme avec les modèles de type GPT), un DataCollatorForLanguageModeling est utilisé afin de préparer dynamiquement les batches d’exemples tokenisés. Le modèle, enrichi par LoRA, est ensuite entraîné sur ce dataset via trainer.train(), avec une gestion automatique des ressources matérielles (utilisation du GPU si disponible), et une sauvegarde automatique du modèle à la fin de chaque époque. La classe Trainer prend en charge l’ensemble de la boucle d’apprentissage (calcul de la perte, mises à jour des paramètres, checkpoints, etc.), rendant le processus d’entraînement à la fois modulaire, simple à mettre en place et reproductible.


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

Dans un premier temps, nous avons réalisé un premier prompt avec une seule question afin de vérifier que le fine-tuning et la sauvegarde avaient bien fonctionné. Nous avons posé une question en lien avec le modèle sur lequel l'entraînement a été effectué.
De plus, nous utilisons à nouveau un pipeline de génération de texte de Hugging Face. Enfin, lors de la génération de la réponse, plusieurs paramètres peuvent être contrôlés. Les plus importants sont, par exemple, le nombre maximal de tokens (ici 80) pour éviter des réponses trop longues, ainsi que la température, qui permet de gérer la créativité des réponses.

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

Maintenant, nous allons comparer les réponses générées par deux modèles : gpt2-medium et gpt2-medium fine-tuné avec LoRA sur le dataset Alpaca. Nous cherchons à savoir si le fine-tuning a été efficace ou non sur les questions issues du dataset Alpaca. Tout d’abord, nous avons posé plusieurs questions provenant du dataset, avec des instructions variées. Ensuite, nous avons à nouveau ajusté les paramètres de génération pour mieux contrôler les réponses. Puis, nous avons chargé les deux modèles que nous souhaitons tester : le modèle fine-tuné et le modèle d’origine. Enfin, nous avons généré les réponses à l’aide du pipeline de Hugging Face, ce qui nous a permis de lancer une comparaison. Pour chaque question, nous avons noté la réponse du modèle gpt2-medium sans fine-tuning, puis celle du modèle fine-tuné, afin d’observer les différences. Nous avons constaté que le modèle de base répond de manière très aléatoire : les mots s’enchaînent sans former de phrases cohérentes, et il arrive qu’il répète plusieurs fois les mêmes mots. En revanche, le modèle fine-tuné est bien plus cohérent : il répond à la question posée, et fournit même parfois des informations supplémentaires pertinentes. Cependant, ses réponses sont parfois trop longues et perdent en clarté. C’est pour cette raison que nous avons limité le nombre maximal de mots par réponse. Nous en concluons que le fine-tuning est fonctionnel, car il produit des réponses pertinentes, claires et utiles dans de nombreuses situations de type question/réponse, comme dans le style du dataset Alpaca.

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

Cette interface interactive permet de répondre aux questions posées sur le dataset Alpaca, en utilisant notre modèle GPT-2 fine-tuné avec LoRA. En effet, une barre de saisie est disponible pour taper votre question (au format du dataset Alpaca) et obtenir une réponse immédiate. Il est possible de poser un grand nombre de questions, et lorsque vous souhaitez arrêter, il suffit de taper "exit" dans la barre. Cette interface permet ainsi une assistance IA facile à utiliser, produisant des réponses claires, pertinentes et de bonne qualité.




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
