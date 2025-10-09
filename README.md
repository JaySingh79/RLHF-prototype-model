**Creating comparison data** - https://colab.research.google.com/github/argilla-io/argilla/blob/v1.11.0/docs/_source/guides/llms/examples/train-reward-model-rlhf.ipynb


**Reward Bench** - https://huggingface.co/papers/2403.13787

**Relevant Resource** : [huyechip article in RLHF](https://huyenchip.com/2023/05/02/rlhf.html)

# Project documentation — Reward-model from synthetic preferences (step-by-step)

This document explains the project end-to-end: what it does, why it is designed that way, how to run it, what each part of the code does, and how to reproduce and evaluate results. It is written as a plain, human-readable guide you can include in your submission. The project implements a small RLHF-style feedback loop: generate or use preference pairs (chosen vs rejected), train a reward model on those preferences, and demonstrate measurable improvement via reranking or lightweight policy updates. The assessment brief this project follows is summarized in the provided assignment. 

---

## 1. High-level overview

Goal

* Demonstrate the core idea of RLHF: learn a reward function from preferences and use it to improve model outputs.
* Use synthetic or public preference data to train a reward model.
* Show before/after metrics and produce artifacts (notebooks, trained checkpoint, Dockerized scoring service, README).

Main steps

1. Prepare a dataset of (prompt, chosen, rejected) preference pairs (synthetic generator or public dataset).
2. Build a reward model that scores (prompt, response) pairs.
3. Train the reward model using a pairwise preference loss.
4. Evaluate the reward model (pairwise accuracy, correlation) and demonstrate improvement by reranking candidate outputs.
5. Save checkpoint(s) and provide a simple API + Dockerfile to serve the model.

---

## 2. Data: how to prepare preference pairs

Accepted formats

* A JSONL file with lines like:

```json
{"prompt":"Why did the service crash?",
 "chosen":"Database connection leak leading to exhaustion of connections.",
 "rejected":"Aliens tampered with the server."}
```

Two practical approaches

* Synthetic generator (recommended to demonstrate understanding):

  * Start from templated prompts and a set of correct answers.
  * Create distractors by lexical swaps, plausible-but-wrong causes, or LLM-generated alternatives.
  * Optionally inject label noise (flip ~5–10% of labels) to mimic realistic human disagreement.
* Public preference datasets (acceptable):

  * Use small subsets of HelpSteer, OpenAssistant or RewardBench if you prefer real annotation. If you use these, document the source and any preprocessing you did.

Suggested dataset size for the internship: 500–2,000 prompts with 2–4 candidates each (gives a few thousand pairwise comparisons). Keep a held-out validation subset (10–20%) for model selection.

---

## 3. Data pipeline: Dataset and collate function

Design choices

* Keep the `Dataset` simple (return raw strings) and perform tokenization in a `collate_fn`. This allows faster batch tokenization and easy tokenizer swaps.

Example `Dataset` (returns raw strings)

```python
class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
```

Example `collate_fn` (batch tokenization)

```python
class PreferenceCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        prompts = [b["prompt"] for b in batch]
        chosen = [b["chosen"] for b in batch]
        rejected = [b["rejected"] for b in batch]
        chosen_enc = self.tokenizer([p+" "+c for p,c in zip(prompts, chosen)],
                                    padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        rejected_enc = self.tokenizer([p+" "+r for p,r in zip(prompts, rejected)],
                                      padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
          "chosen_input_ids": chosen_enc["input_ids"],
          "chosen_mask": chosen_enc["attention_mask"],
          "rejected_input_ids": rejected_enc["input_ids"],
          "rejected_mask": rejected_enc["attention_mask"]
        }
```

DataLoader example

```python
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True)
```

---

## 4. Model: reward model architecture

Goal

* Map (prompt, response) → scalar reward.

Recommended architecture

* Encoder backbone (pretrained transformer encoder, e.g. `distilbert-base-uncased`).
* Simple head: a linear layer on top of the [CLS] embedding that outputs a single scalar (no activation).

Example class

```python
class RewardModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, 1)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.last_hidden_state[:,0,:]
        score = self.head(cls).squeeze(-1)
        return score
```

Notes

* The reward model does not generate text.
* You can freeze most encoder layers to speed up training (see optimizations below). Optionally unfreeze top layers later if needed.

---

## 5. Loss function: pairwise preference loss

Core loss

* For a pair (chosen, rejected), the model produces two scalars `s_chosen` and `s_rejected`. Use the logistic pairwise loss:

```
L = -log(sigmoid(s_chosen - s_rejected))
```

* This encourages `s_chosen > s_rejected`.

Implementation

```python
def preference_loss(chosen_scores, rejected_scores):
    return -torch.log(torch.sigmoid(chosen_scores - rejected_scores)).mean()
```

This is the Bradley–Terry / pairwise logistic loss; DPO, PPO preference losses are built on similar pairwise or preference-derived objectives.

---

## 6. Training loop (optimized, ready-to-run)

Optimizations included

* Freeze encoder parameters initially (train head only).
* Mixed precision (AMP) for GPUs.
* Learning rate scheduler with warmup.
* tqdm progress bar and average loss reporting.
* DataLoader `num_workers` and `pin_memory` settings.

Core training code

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = RewardModel().to(device)

# freeze encoder (optional)
for p in model.encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
num_epochs = 3
num_training_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            chosen_scores = model(batch["chosen_input_ids"].to(device), batch["chosen_mask"].to(device))
            rejected_scores = model(batch["rejected_input_ids"].to(device), batch["rejected_mask"].to(device))
            loss = preference_loss(chosen_scores, rejected_scores)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} average loss: {epoch_loss/len(dataloader):.4f}")
    # save checkpoint when desirable
    torch.save(model.state_dict(), f"checkpoints/rm_epoch{epoch+1}.pt")
```

Notes

* If you do not have a GPU, remove AMP and use a smaller batch size.
* To fine-tune the encoder later, unfreeze top N layers (or all layers) and continue training with a lower learning rate.

---

## 7. Evaluation: metrics to compute and how

Primary evaluation goals

1. Measure how well the reward model reproduces labeled preferences (pairwise accuracy).
2. Demonstrate that using the reward model improves selection of answers (selection accuracy / win-rate).
3. Show correlation between reward scores and ground-truth quality (Spearman rank correlation), optionally report loss curves.

How to compute pairwise accuracy

* For each validation pair, compute `s_chosen` and `s_rejected`. If `s_chosen > s_rejected`, that counts as a correct prediction.

```python
correct = (chosen_scores > rejected_scores).sum().item()
accuracy = correct / len(validation_pairs)
```

How to compute selection accuracy (reranking demo)

* For each prompt in test set, obtain K candidate responses (either from your synthetic dataset or by sampling from the base generator).
* Baseline selection: choose the first / highest-prob candidate.
* Reranking selection: compute reward scores for each candidate and choose the argmax.
* Compute fraction where chosen candidate equals ground-truth best (or compute pairwise comparisons vs ground-truth best).
* Report delta: rerank_accuracy - baseline_accuracy, and report counts and confidence intervals if you want statistical rigor.

Spearman correlation

* For prompts with more than two candidates and a ground-truth ranking (or similarity scores), compute Spearman rank correlation between reward scores and ground-truth ranking.

Logging

* Log per-epoch training & validation loss, pairwise accuracy, and examples of before/after selections (qualitative). A short table with 10 examples of baseline vs rerank output is powerful.

---

## 8. Checkpointing and reproducibility

What to save

* Model weights (`model.save_pretrained` or `torch.save(state_dict)`).
* Tokenizer artifacts.
* Training config (hyperparameters, seed, optimizer, scheduler settings) as `config.json`.
* A small sample of the dataset used (train/val split) and the exact random seed.

Reproducibility tips

* Set seeds for `random`, `numpy`, and `torch`.
* Record the environment (Python version, PyTorch/transformers versions).
* Provide a `requirements.txt` and a `run.sh` script that reproduces basic steps.

---

## 9. Serving the model and Dockerization

Serving

* Provide a small FastAPI (or Flask) app that loads the reward model and exposes a `/score` endpoint that accepts `prompt` and `response` and returns a scalar score.

Example `serve.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import torch

class Req(BaseModel):
    prompt: str
    response: str

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = RewardModel()
model.load_state_dict(torch.load("checkpoints/rm_best.pt", map_location="cpu"))
model.eval()

@app.post("/score")
def score(req: Req):
    enc = tokenizer(req.prompt + " " + req.response, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        s = model(enc["input_ids"], enc["attention_mask"]).item()
    return {"score": float(s)}
```

## 10. Suggested notebooks and what to include

0_data_generation.ipynb

* Show how you generated synthetic feedback (templates or LLM prompts).
* Print sample prompt/candidate pairs and label distribution.

1_train_reward_model.ipynb

* Dataset loading and collate usage.
* Model definition.
* Training loop (with progress bars) and training curves.
* Save best checkpoint.

2_evaluate_and_demo.ipynb

* Pairwise accuracy and validation metrics.
* Reranking demo: show before/after selections for example prompts.
* Optional: reward-weighted fine-tuning or a short DPO example.

---


## 15. Short example of key code snippets (for copy/paste)

Preference loss:

```python
def preference_loss(chosen_scores, rejected_scores):
    return -torch.log(torch.sigmoid(chosen_scores - rejected_scores)).mean()
```

Training checkpoint save:

```python
torch.save({"model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()},
           "checkpoints/rm_epochX.pt")
```

Pairwise accuracy:

```python
with torch.no_grad():
    chosen_s = model(chosen_input_ids, chosen_mask)
    rejected_s = model(rejected_input_ids, rejected_mask)
acc = (chosen_s > rejected_s).float().mean().item()
```

---
