import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

class RewardModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden_size, 1)  # scalar reward

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        reward = self.head(cls_emb).squeeze(-1)       # (batch,)
        return reward


# 2. Dummy Dataset (prompt, winning_response, losing_response)
class PreferenceDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt, winning, losing = item["prompt"], item["winning"], item["losing"]

        # Encode winning and losing separately
        winning_enc = self.tokenizer(prompt + " " + winning,
                                    truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        losing_enc = self.tokenizer(prompt + " " + losing,
                                      truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")

        return {
            "winning_input_ids": winning_enc["input_ids"].squeeze(),
            "winning_mask": winning_enc["attention_mask"].squeeze(),
            "losing_input_ids": losing_enc["input_ids"].squeeze(),
            "losing_mask": losing_enc["attention_mask"].squeeze(),
        }


# Pairwise Loss
def preference_loss(winning_scores, losing_scores):
    # Bradley–Terry / log loss: -log σ(s_winning - s_losing)
    return -torch.log(torch.sigmoid(winning_scores - losing_scores)).mean()
