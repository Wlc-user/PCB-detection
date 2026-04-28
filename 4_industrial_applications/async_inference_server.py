import asyncio, torch, time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
import uvicorn

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)
model_pool = None

@app.on_event("startup")
async def startup():
    global model_pool
    model_pool = [torch.nn.Linear(100, 10) for _ in range(2)]
    for m in model_pool:
        m(torch.randn(1, 100))
    print("models ready")

@app.post("/infer")
async def infer(data: list):
    t0 = time.time()
    tensor = torch.tensor(data).float()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, lambda: model_pool[0](tensor).detach().numpy()
    )
    return {
        "result": result.tolist(),
        "latency_ms": round((time.time() - t0) * 1000, 2)
    }

@app.get("/health")
async def health():
    return {"status": "ok"}