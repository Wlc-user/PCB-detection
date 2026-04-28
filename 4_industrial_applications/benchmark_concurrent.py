import asyncio, aiohttp, time, numpy as np

async def send_request(session, url, data):
    async with session.post(url, json={"data": data}) as resp:
        return await resp.json()

async def benchmark(n=100, c=10):
    url = "http://localhost:8001/infer"
    data = [1.0] * 100
    connector = aiohttp.TCPConnector(limit=c)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()
        tasks = [send_request(session, url, data) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
    lats = [r["latency_ms"] for r in results]
    print(f"QPS:{n/elapsed:.0f} Avg:{np.mean(lats):.1f}ms P99:{np.percentile(lats,99):.1f}ms")

if __name__ == "__main__":
    asyncio.run(benchmark(100, 10))