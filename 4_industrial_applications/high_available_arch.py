"""
高并发高可用流式系统架构
 High Concurrency & High Availability Streaming System

核心策略:
    1. 水平扩展 - 多Worker实例
    2. 负载均衡 - Nginx / K8s
    3. 服务降级 - 熔断 + 限流
    4. 数据分区 - Redis Cluster
    5. 异步非阻塞 - asyncio + 协程池

架构图:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         负载均衡层                               │
    │   ┌─────────┐  ┌─────────┐  ┌─────────┐                        │
    │   │ Nginx   │  │ K8s LB  │  │ Consul  │  ◄── 健康检查         │
    │   │ upstream│  │ Service │  │ 选举    │                        │
    │   └────┬────┘  └────┬────┘  └────┬────┘                        │
    └─────────┼───────────┼───────────┼─────────────────────────────┘
               │           │           │
               ▼           ▼           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                         边缘节点层                               │
    │   ┌─────────┐  ┌─────────┐  ┌─────────┐                        │
    │   │ Edge #1 │  │ Edge #2 │  │ Edge #3 │  ◄── RTSP拉流         │
    │   │ Worker  │  │ Worker  │  │ Worker  │     本地推理          │
    │   └────┬────┘  └────┬────┘  └────┬────┘                        │
    └─────────┼───────────┼───────────┼─────────────────────────────┘
              │           │           │
              └───────────┴───────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                         消息队列层                               │
    │   ┌─────────┐  ┌─────────┐  ┌─────────┐                        │
    │   │ Kafka   │  │ Redis   │  │ RabbitMQ│                        │
    │   │ Streams │  │ Streams │  │ Streams │  ◄── 事件总线          │
    │   └─────────┘  └─────────┘  └─────────┘                        │
    └─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                         AI推理层                                │
    │   ┌─────────┐  ┌─────────┐  ┌─────────┐                        │
    │   │ GPU #1  │  │ GPU #2  │  │ GPU #3  │  ◄── YOLO/LLM          │
    │   │ Batch   │  │ Batch   │  │ Batch   │     TensorRT           │
    │   └─────────┘  └─────────┘  └─────────┘                        │
    └─────────────────────────────────────────────────────────────────┘
"""

import os
import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# ============ 日志配置 ============
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('HighAvailability')


# ============================================================================
# 1. 限流器 - 令牌桶算法
# ============================================================================
class RateLimiter:
    """令牌桶限流器"""
    
    def __init__(self, rate: int, capacity: int):
        self.rate = rate  # 每秒令牌数
        self.capacity = capacity  # 桶容量
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def allow(self, tokens: int = 1) -> bool:
        """检查是否允许请求"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # 补充令牌
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def async_allow(self, tokens: int = 1) -> bool:
        """异步版本"""
        return self.allow(tokens)


class SlidingWindowRateLimiter:
    """滑动窗口限流器"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = threading.Lock()
    
    def allow(self) -> bool:
        with self.lock:
            now = time.time()
            
            # 清理过期请求
            cutoff = now - self.window_seconds
            self.requests = [t for t in self.requests if t > cutoff]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False


# ============================================================================
# 2. 熔断器 - Circuit Breaker
# ============================================================================
class CircuitBreakerState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        """执行带熔断保护的调用"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("[CircuitBreaker] State: HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("[CircuitBreaker] State: CLOSED")
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"[CircuitBreaker] State: OPEN (failures={self.failure_count})")
            
            raise e


# ============================================================================
# 3. 连接池
# ============================================================================
class ConnectionPool:
    """连接池"""
    
    def __init__(self, factory: Callable, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self.pool = queue.Queue(maxsize=max_size)
        self.size = 0
        self.lock = threading.Lock()
    
    def get(self, timeout: float = 5.0):
        """获取连接"""
        try:
            return self.pool.get(timeout=timeout)
        except queue.Empty:
            with self.lock:
                if self.size < self.max_size:
                    self.size += 1
                    return self.factory()
            raise Exception("Connection pool exhausted")
    
    def put(self, conn):
        """归还连接"""
        try:
            self.pool.put_nowait(conn)
        except queue.Full:
            with self.lock:
                self.size -= 1


class AsyncConnectionPool:
    """异步连接池"""
    
    def __init__(self, factory: Callable, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self.pool = asyncio.Queue(maxsize=max_size)
        self.size = 0
    
    async def get(self):
        """获取连接"""
        try:
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            if self.size < self.max_size:
                self.size += 1
                return await self.factory()
            return await self.pool.get()
    
    async def put(self, conn):
        """归还连接"""
        try:
            self.pool.put_nowait(conn)
        except asyncio.QueueFull:
            pass


# ============================================================================
# 4. 分布式缓存
# ============================================================================
class DistributedCache:
    """分布式缓存"""
    
    def __init__(self, redis_url: str = None):
        self.redis = None
        self.local_cache = {}  # 本地L1缓存
        self.local_ttl = 60    # 本地缓存TTL
        self.hits = 0
        self.misses = 0
        
        if redis_url:
            try:
                import redis
                self.redis = redis.from_url(redis_url)
                logger.info("[Cache] Redis connected")
            except ImportError:
                logger.warning("[Cache] Redis not available, using local cache")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # L1: 本地缓存
        if key in self.local_cache:
            entry = self.local_cache[key]
            if time.time() - entry['ts'] < self.local_ttl:
                self.hits += 1
                return entry['value']
            else:
                del self.local_cache[key]
        
        # L2: Redis缓存
        if self.redis:
            try:
                value = self.redis.get(key)
                if value:
                    self.hits += 1
                    self.local_cache[key] = {'value': value, 'ts': time.time()}
                    return value
            except:
                pass
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        self.local_cache[key] = {'value': value, 'ts': time.time()}
        
        if self.redis:
            try:
                self.redis.setex(key, ttl, value)
            except:
                pass
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2%}",
            'local_size': len(self.local_cache)
        }


# ============================================================================
# 5. 消息队列
# ============================================================================
class MessageQueue:
    """消息队列"""
    
    def __init__(self, broker: str = 'redis'):
        self.broker = broker
        self.queue = asyncio.Queue(maxsize=1000)
        self.publisher = None
        self.subscribers: List[asyncio.Queue] = []
        
        self.stats = {'published': 0, 'consumed': 0, 'dropped': 0}
    
    async def publish(self, topic: str, message: Any):
        """发布消息"""
        try:
            self.queue.put_nowait((topic, message))
            self.stats['published'] += 1
        except asyncio.QueueFull:
            self.stats['dropped'] += 1
            logger.warning(f"[MQ] Queue full, dropping message")
    
    async def subscribe(self, topic: str = None) -> Any:
        """订阅消息"""
        msg_topic, message = await self.queue.get()
        
        if topic is None or topic == msg_topic:
            self.stats['consumed'] += 1
            return message
        else:
            # 放回队列
            await self.queue.put((msg_topic, message))
            return await self.subscribe(topic)
    
    def get_stats(self) -> Dict:
        return self.stats


# ============================================================================
# 6. 负载均衡器
# ============================================================================
class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: str = 'round_robin'):
        self.strategy = strategy
        self.backends: List[Dict] = []
        self.current_index = 0
        self.lock = threading.Lock()
        
        # 健康检查
        self.health_check_interval = 30
        self.last_health_check = time.time()
    
    def add_backend(self, url: str, weight: int = 1):
        """添加后端"""
        self.backends.append({
            'url': url,
            'weight': weight,
            'healthy': True,
            'active_requests': 0,
            'total_requests': 0,
            'failures': 0
        })
    
    def get_backend(self) -> Optional[str]:
        """获取后端"""
        with self.lock:
            healthy_backends = [b for b in self.backends if b['healthy']]
            
            if not healthy_backends:
                logger.warning("[LB] No healthy backends")
                return None
            
            if self.strategy == 'round_robin':
                backend = healthy_backends[self.current_index % len(healthy_backends)]
                self.current_index += 1
            
            elif self.strategy == 'least_connections':
                backend = min(healthy_backends, key=lambda x: x['active_requests'])
            
            elif self.strategy == 'weighted':
                # 加权随机
                total_weight = sum(b['weight'] for b in healthy_backends)
                r = sum(b['weight'] for b in healthy_backends if 
                       sum(b2['weight'] for b2 in healthy_backends[:healthy_backends.index(b)]) < 
                       (r := __import__('random').random() * total_weight))
                backend = healthy_backends[r]
            
            backend['active_requests'] += 1
            backend['total_requests'] += 1
            
            return backend['url']
    
    def release_backend(self, url: str, success: bool = True):
        """释放后端"""
        with self.lock:
            for b in self.backends:
                if b['url'] == url:
                    b['active_requests'] = max(0, b['active_requests'] - 1)
                    if not success:
                        b['failures'] += 1
                        # 失败率过高标记为不健康
                        if b['failures'] > 10 and b['failures'] / b['total_requests'] > 0.5:
                            b['healthy'] = False
                            logger.warning(f"[LB] Backend unhealthy: {url}")
                    break


# ============================================================================
# 7. Worker池
# ============================================================================
class WorkerPool:
    """Worker池"""
    
    def __init__(self, max_workers: int = None, queue_size: int = 1000):
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        self.max_workers = max_workers
        self.task_queue = asyncio.Queue(maxsize=queue_size)
        self.result_queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        # 统计
        self.stats = {'queued': 0, 'completed': 0, 'failed': 0}
    
    async def worker(self, worker_id: int):
        """Worker协程"""
        logger.info(f"[Worker-{worker_id}] Started")
        
        while self.running:
            try:
                # 带超时的获取
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                func, args, kwargs = task
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    await self.result_queue.put(('success', result))
                    self.stats['completed'] += 1
                    
                except Exception as e:
                    await self.result_queue.put(('error', str(e)))
                    self.stats['failed'] += 1
                
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] Error: {e}")
        
        logger.info(f"[Worker-{worker_id}] Stopped")
    
    async def start(self):
        """启动Worker池"""
        self.running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self.worker(i))
            self.workers.append(worker)
        
        logger.info(f"[WorkerPool] Started {self.max_workers} workers")
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """提交任务"""
        await self.task_queue.put((func, args, kwargs))
        self.stats['queued'] += 1
        
        return await self.result_queue.get()
    
    async def stop(self):
        """停止Worker池"""
        self.running = False
        
        for w in self.workers:
            w.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info(f"[WorkerPool] Stopped. Stats: {self.stats}")


# ============================================================================
# 8. 高可用服务
# ============================================================================
class HighAvailabilityService:
    """高可用服务"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 限流
        self.rate_limiter = RateLimiter(
            rate=config.get('rate_limit', 100),
            capacity=config.get('burst_size', 200)
        )
        
        # 熔断
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('failure_threshold', 5),
            recovery_timeout=config.get('recovery_timeout', 60)
        )
        
        # 缓存
        self.cache = DistributedCache(
            redis_url=config.get('redis_url')
        )
        
        # 消息队列
        self.mq = MessageQueue()
        
        # Worker池
        self.worker_pool = WorkerPool(
            max_workers=config.get('max_workers', mp.cpu_count())
        )
        
        # 负载均衡
        self.load_balancer = LoadBalancer(
            strategy=config.get('lb_strategy', 'round_robin')
        )
        
        # 健康检查
        self.health_check_interval = config.get('health_check_interval', 30)
        self.is_healthy = True
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rejected_requests': 0,
            'avg_latency': 0
        }
    
    async def start(self):
        """启动服务"""
        await self.worker_pool.start()
        logger.info("[HAService] Started")
    
    async def stop(self):
        """停止服务"""
        await self.worker_pool.stop()
        logger.info("[HAService] Stopped")
    
    async def handle_request(self, request: Dict) -> Dict:
        """处理请求"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # 1. 限流检查
        if not self.rate_limiter.allow():
            self.stats['rejected_requests'] += 1
            return {'error': 'Rate limit exceeded', 'status': 429}
        
        # 2. 检查健康状态
        if not self.is_healthy:
            return {'error': 'Service unavailable', 'status': 503}
        
        # 3. 缓存检查
        cache_key = self._generate_cache_key(request)
        cached = self.cache.get(cache_key)
        if cached:
            return {'cached': True, 'data': cached}
        
        try:
            # 4. 执行处理
            result = await self._process_request(request)
            
            # 5. 更新统计
            self.stats['successful_requests'] += 1
            latency = time.time() - start_time
            self._update_latency_stats(latency)
            
            # 6. 缓存结果
            self.cache.set(cache_key, result)
            
            return {'data': result, 'latency': latency}
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"[HAService] Request failed: {e}")
            return {'error': str(e), 'status': 500}
    
    async def _process_request(self, request: Dict) -> Any:
        """处理请求核心逻辑"""
        # 使用Worker池处理
        result, _ = await self.worker_pool.submit(self._process_task, request)
        return result
    
    def _process_task(self, request: Dict) -> Any:
        """实际处理任务"""
        # 模拟处理
        time.sleep(0.01)
        return {'result': 'processed', 'request_id': request.get('id')}
    
    def _generate_cache_key(self, request: Dict) -> str:
        """生成缓存键"""
        content = json.dumps(request, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _update_latency_stats(self, latency: float):
        """更新延迟统计"""
        n = self.stats['total_requests']
        old_avg = self.stats['avg_latency']
        self.stats['avg_latency'] = (old_avg * (n - 1) + latency) / n
    
    async def health_check(self):
        """健康检查"""
        # 检查所有依赖服务
        checks = {
            'cache': self.cache.redis is not None if hasattr(self.cache, 'redis') else True,
            'worker_pool': len(self.worker_pool.workers) > 0,
        }
        
        self.is_healthy = all(checks.values())
        
        return {
            'healthy': self.is_healthy,
            'checks': checks,
            'stats': self.stats
        }
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'cache_stats': self.cache.get_stats(),
            'mq_stats': self.mq.get_stats(),
            'healthy': self.is_healthy
        }


# ============================================================================
# 9. 流式处理管道
# ============================================================================
class StreamingPipeline:
    """流式处理管道"""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.processors: List[Callable] = []
        self.running = False
        
        # 背压控制
        self.input_queue = asyncio.Queue(maxsize=buffer_size)
        self.output_queue = asyncio.Queue(maxsize=buffer_size)
        
        # 统计
        self.stats = {'in': 0, 'out': 0, 'dropped': 0}
    
    def add_processor(self, processor: Callable):
        """添加处理器"""
        self.processors.append(processor)
    
    async def start(self):
        """启动管道"""
        self.running = True
        self.pipeline_task = asyncio.create_task(self._run_pipeline())
        logger.info(f"[Pipeline] Started with {len(self.processors)} processors")
    
    async def _run_pipeline(self):
        """运行管道"""
        while self.running:
            try:
                # 获取输入
                item = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)
                self.stats['in'] += 1
                
                # 管道处理
                result = item
                for processor in self.processors:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(result)
                    else:
                        result = processor(result)
                
                # 输出
                try:
                    self.output_queue.put_nowait(result)
                    self.stats['out'] += 1
                except asyncio.QueueFull:
                    self.stats['dropped'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[Pipeline] Error: {e}")
    
    async def push(self, item: Any):
        """推入数据"""
        try:
            self.input_queue.put_nowait(item)
        except asyncio.QueueFull:
            self.stats['dropped'] += 1
    
    async def pull(self, timeout: float = 1.0) -> Optional[Any]:
        """拉取数据"""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def stop(self):
        """停止管道"""
        self.running = False
        if hasattr(self, 'pipeline_task'):
            self.pipeline_task.cancel()


# ============================================================================
# 演示
# ============================================================================
async def demo_high_availability():
    """高并发高可用演示"""
    print("=" * 60)
    print(" High Concurrency & HA Demo")
    print("=" * 60)
    
    # 1. 限流器演示
    print("\n[1] Rate Limiter Demo")
    limiter = RateLimiter(rate=10, capacity=20)
    
    for i in range(25):
        allowed = limiter.allow()
        status = "[OK]" if allowed else "[X]"
        print(f"   Request {i+1}: {status} (tokens={limiter.tokens:.1f})")
        await asyncio.sleep(0.05)
    
    # 2. 熔断器演示
    print("\n[2] Circuit Breaker Demo")
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    def fail_task():
        raise Exception("Task failed")
    
    for i in range(6):
        try:
            cb.call(fail_task)
        except Exception as e:
            print(f"   Call {i+1}: {e} (state={cb.state.value})")
    
    # 3. 缓存演示
    print("\n[3] Cache Demo")
    cache = DistributedCache()
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    
    print(f"   get(key1): {cache.get('key1')}")
    print(f"   get(key2): {cache.get('key2')}")
    print(f"   get(key3): {cache.get('key3')}")
    print(f"   Stats: {cache.get_stats()}")
    
    # 4. Worker池演示
    print("\n[4] Worker Pool Demo")
    pool = WorkerPool(max_workers=4)
    await pool.start()
    
    async def sample_task(x):
        await asyncio.sleep(0.1)
        return x * 2
    
    tasks = []
    for i in range(8):
        result, _ = await pool.submit(sample_task, i)
        tasks.append(result)
    
    print(f"   Results: {tasks}")
    await pool.stop()
    
    # 5. 流式管道演示
    print("\n[5] Streaming Pipeline Demo")
    pipeline = StreamingPipeline(buffer_size=50)
    
    # 添加处理器
    pipeline.add_processor(lambda x: x * 2)
    pipeline.add_processor(lambda x: x + 1)
    pipeline.add_processor(lambda x: f"Result: {x}")
    
    await pipeline.start()
    
    # 发送数据
    for i in range(5):
        await pipeline.push(i)
        result = await pipeline.pull()
        print(f"   Input: {i} -> Output: {result}")
    
    await pipeline.stop()
    
    print("\n" + "=" * 60)
    print(" Demo Complete!")
    print("=" * 60)


# ============================================================================
# 主函数
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='High Availability Streaming System')
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'service'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rate-limit', type=int, default=100)
    parser.add_argument('--redis-url', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        asyncio.run(demo_high_availability())
    else:
        config = {
            'max_workers': args.workers,
            'rate_limit': args.rate_limit,
            'redis_url': args.redis_url,
        }
        service = HighAvailabilityService(config)
        asyncio.run(service.start())


"""
==============================================================================
高并发高可用架构总结
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                              核心策略                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 限流 (Rate Limiting)                                                   │
│     ├── 令牌桶: burst流量处理                                               │
│     ├── 滑动窗口: 精确控制                                                  │
│     └── 分布式限流: Redis + Lua                                            │
│                                                                             │
│  2. 熔断 (Circuit Breaker)                                                 │
│     ├── Closed: 正常 -> 失败累积                                           │
│     ├── Open: 快速失败, 等待恢复                                             │
│     └── Half-Open: 探测恢复                                                │
│                                                                             │
│  3. 降级 (Graceful Degradation)                                            │
│     ├── 功能降级: 非核心功能关闭                                            │
│     ├── 数据降级: 返回缓存/默认值                                           │
│     └── 服务降级: 延迟响应                                                  │
│                                                                             │
│  4. 水平扩展 (Horizontal Scaling)                                           │
│     ├── 无状态服务: 多实例无差别                                           │
│     ├── K8s HPA: 自动扩缩容                                                │
│     └── 区域分布: 多AZ部署                                                 │
│                                                                             │
│  5. 负载均衡 (Load Balancing)                                              │
│     ├── 轮询: 简单公平                                                      │
│     ├── 最少连接: 优化资源                                                  │
│     └── 加权: 异构系统                                                      │
│                                                                             │
│  6. 缓存 (Caching)                                                         │
│     ├── L1: 本地内存 (ns级)                                                │
│     ├── L2: Redis (μs级)                                                  │
│     └── 读写分离: 热点分离                                                  │
│                                                                             │
│  7. 异步处理 (Async Processing)                                             │
│     ├── asyncio: IO密集型                                                  │
│     ├── Worker Pool: CPU密集型                                             │
│     └── 消息队列: 生产/消费分离                                            │
│                                                                             │
│  8. 健康检查 (Health Check)                                                │
│     ├── 主动探测: 定期检查                                                 │
│     ├── 被动检测: 请求失败率                                               │
│     └── 自动剔除: 不健康节点                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

部署架构:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                        ┌─────────────┐                                      │
│                        │   Client   │                                      │
│                        └──────┬──────┘                                      │
│                               │                                              │
│                        ┌──────▼──────┐                                      │
│                        │    Nginx   │ ◄── SSL终止, 静态资源                │
│                        │   (LB)     │ ◄── 健康检查, 限流                    │
│                        └──────┬──────┘                                      │
│                               │                                              │
│              ┌────────────────┼────────────────┐                           │
│              │                │                │                           │
│        ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐                       │
│        │  API #1   │    │  API #2   │    │  API #3   │                       │
│        │ (Worker)  │    │ (Worker)  │    │ (Worker)  │                       │
│        └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                       │
│              │                │                │                            │
│              └────────────────┼────────────────┘                            │
│                               │                                              │
│                        ┌──────▼──────┐                                      │
│                        │    Kafka    │ ◄── 消息队列                         │
│                        │   (Queue)   │ ◄── 异步处理                         │
│                        └──────┬──────┘                                      │
│                               │                                              │
│              ┌────────────────┼────────────────┐                           │
│              │                │                │                            │
│        ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐                       │
│        │  GPU #1   │    │  GPU #2   │    │  GPU #3   │                       │
│        │ (YOLO)    │    │ (YOLO)    │    │ (YOLO)    │                       │
│        └───────────┘    └───────────┘    └───────────┘                       │
│                               │                                              │
│                        ┌──────▼──────┐                                      │
│                        │   Redis     │ ◄── 结果缓存, 会话                    │
│                        │  Cluster    │ ◄── 发布/订阅                         │
│                        └─────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

性能指标目标:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  指标                    目标值              实际值                           │
│  ─────────────────────────────────────────────────────────────────────    │
│  QPS (Queries/Sec)     10,000+            根据实例数线性扩展                  │
│  Latency P99           < 100ms           目标99线                           │
│  Availability           99.99%             多AZ部署                          │
│  Error Rate            < 0.01%           熔断保护                          │
│  CPU Utilization       60-80%            自动扩缩触发点                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
