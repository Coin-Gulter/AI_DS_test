from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("conll2003")

# Shrink dataset for faster training
shrink_data_factor = 0.1
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * shrink_data_factor)))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(int(len(dataset["validation"]) * shrink_data_factor)))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(int(len(dataset["test"]) * shrink_data_factor)))

# Define list of common animals
ANIMALS = ["butterfly", "cat", "cow", "dog", "elephant", "horse", "chicken", "sheep", "spider", "squirrel"]

MODEL_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3)  # 3 labels: B-Animal, I-Animal, O

# Tokenization Function with Animal Labels
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=64)
    labels = []
    for i, words in enumerate(batch["tokens"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif words[word_idx].lower() in ANIMALS:
                if previous_word_idx != word_idx:
                    label_ids.append(1)  # B-Animal
                else:
                    label_ids.append(2)  # I-Animal
            else:
                label_ids.append(0)  # O (not an entity)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == "__main__":
    print("Filtering dataset to retain only animal entities...")
    dataset = dataset.map(tokenize_and_align_labels, batched=True)

    training_args = TrainingArguments(
        output_dir="./models/ner",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("Training NER model for animal recognition...")
    trainer.train()

    print("Saving trained model...")
    model.save_pretrained("./models/ner/weights")
    tokenizer.save_pretrained("./models/ner/weights")
    print("Training complete! Model saved.")
