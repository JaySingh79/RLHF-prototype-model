from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_untrained = RewardModel(model_name).to(device)
    model = RewardModel(model_name).to(device)

    train_dataset = PreferenceDataset(tokenizer, final_dataset)
    trained_model, evaluator = train_with_evaluation(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        num_epochs=5,
        batch_size=2, 
        learning_rate=1e-5
    )
    
    print("\n" + "="*60)
    print("GENERATING EVALUATION PLOTS")
    print("="*60)
    
    evaluator.plot_training_progress()
    evaluator.plot_score_distribution(test_data)
    
    demonstrate_model_improvements(
        model_before=model_untrained,
        model_after=trained_model,
        test_samples=test_data,
        tokenizer=tokenizer,
        device=device
    )
    
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    
    final_metrics = evaluator.evaluate_on_pairs(test_data)
    print(f"Final Model Performance:")
    print(f"  - Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  - AUC-ROC: {final_metrics['auc']:.4f}")
    print(f"  - Average Margin: {final_metrics['avg_margin']:.4f}")
    print(f"  - Consistency: {final_metrics['consistency']:.4f}")
    
    torch.save(trained_model.state_dict(), 'reward_model_trained.pt')
    print("\n✓ Model saved as 'reward_model_trained.pt'")
    with open('training_history.json', 'w') as f:
        json.dump(evaluator.history, f, indent=2)
    print("Training history saved as 'training_history.json'")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    model_untrained = RewardModel("distilbert-base-uncased").to(device)
    model = RewardModel("distilbert-base-uncased").to(device)
    train_dataset = PreferenceDataset(tokenizer, final_dataset)    
    trained_model, evaluator = train_with_evaluation(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        num_epochs=3,
        batch_size=20,
        learning_rate=1e-5
    )
    
    evaluator.plot_training_progress()
    evaluator.plot_score_distribution(test_dataset)
    
    demonstrate_model_improvements(
        model_before=model_untrained,
        model_after=trained_model,
        test_samples=test_dataset[:10],
        tokenizer=tokenizer,
        device=device
    )
    
    print("\nTraining and evaluation completed successfully!")