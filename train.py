import argparse
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
import torch
import time

model_map = {
    "bert": "google-bert/bert-base-uncased",
    "roberta": "FacebookAI/roberta-base"
}

dataset_map = {
    "sst2": "stanfordnlp/sst2",
    "imdb": "stanfordnlp/imdb"
}

def compute_metrics(eval_pred):
    """Computes accuracy and F1 score for the trainer."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    
    return {**acc, **f1}

def get_dataset(dataset_name):
    """Loads and returns the specific dataset splits."""
    print(f"Loading {dataset_name} dataset...")
    raw_dataset = load_dataset(dataset_map[dataset_name])
    train_dataset = raw_dataset["train"]
    test_dataset = raw_dataset["test"]

    if dataset_name.lower() == "sst2":
        train_dataset = train_dataset.rename_column("sentence", "text")
        test_dataset = test_dataset.rename_column("sentence", "text")

    return train_dataset, test_dataset

def print_config(args):
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("\n")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default="bert", 
        help="Model identifier (e.g., bert, roberta)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="sst2", 
        help="Dataset name (sst2 or imdb)"
    )
    parser.add_argument(
        "--peft", 
        type=str, 
        default="none", 
        help="PEFT method (e.g., none, lora, prefix)"
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=8,
        help="Rank for PEFT methods like LoRA"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5, 
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-6, 
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="Max token length"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    args = parser.parse_args()
    assert args.model in model_map, f"Model {args.model} not supported."
    assert args.dataset in dataset_map, f"Dataset {args.dataset} not supported."

    print_config(args)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Dataset
    train_dataset, test_dataset = get_dataset(args.dataset)
    
    # Optional: limit dataset size for debugging/fast testing
    # train_dataset = train_dataset.select(range(5000)) 
    # test_dataset = test_dataset.select(range(500))

    # 2. Load Tokenizer
    model_id = model_map[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 3. Preprocess Data
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = test_dataset.map(tokenize_function, batched=True)

    # 4. Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=2
    )
    model.to(device)

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results_{args.model_name}_{args.dataset}",
        eval_strategy="epoch",            # Evaluate at the end of every epoch
        save_strategy="epoch",            # Save checkpoint at the end of every epoch
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,      # Load the best model when finished
        metric_for_best_model="accuracy",
        logging_dir='./logs',
        logging_steps=50,
        report_to="none"                  # Set to "wandb" if you use Weights & Biases
    )

    # 6. Initialize Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    # 8. Evaluate
    eval_results = trainer.evaluate()
    
    print_config(args)
    print(f"Using device: {device}")
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    print(f"\nFinal Evaluation Results for {args.model_name} on {args.dataset}:")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"Loss:     {eval_results['eval_loss']:.4f}")

if __name__ == "__main__":
    main()