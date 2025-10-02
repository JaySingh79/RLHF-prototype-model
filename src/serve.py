from fastapi import FastAPI
from transformers import AutoTokenizer
from reward_model import RewardModel
import torch

app = FastAPI()
model = RewardModel("checkpoint_dir")
tokenizer = AutoTokenizer.from_pretrained("checkpoint_dir")
model.eval()

@app.post("/score")
def score(prompt: str, response: str):
    inputs = tokenizer(prompt + " [SEP] " + response, return_tensors="pt", truncation=True)
    with torch.no_grad():
        s = model(**inputs).item()
    return {"score": s}