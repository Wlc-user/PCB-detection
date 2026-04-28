from celery import Celery
import torch

app = Celery("inference", broker="redis://localhost:6379/0")
model_cache = {}

def load_model(device="cpu"):
    if device not in model_cache:
        model_cache[device] = torch.nn.Linear(100, 10).to(device)
        model_cache[device].eval()
    return model_cache[device]

@app.task
def infer_task(data, device="cpu"):
    model = load_model(device)
    tensor = torch.tensor(data).float().to(device)
    with torch.no_grad():
        return model(tensor).cpu().numpy().tolist()