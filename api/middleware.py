import time
import os
import psutil
import redis.asyncio as aioredis
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        current_second = int(start_time)
        key = f"rate_limit:{current_second}"
        
        req_count = await redis_client.incr(key)
        
        if req_count == 1:
            await redis_client.expire(key, 5)
            
        limit_str = await redis_client.get("global_ai_limit")
        limit = int(limit_str) if limit_str else 10000
        
        if req_count > limit:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
            
        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000
        
        # CHANGED: status_code + error_flag replaces mem_usage
        cpu_usage = psutil.cpu_percent(interval=None)
        status_code = response.status_code
        error_flag = 1 if status_code >= 400 else 0
        
        log_line = f"{start_time},{req_count},{process_time:.2f},{cpu_usage},{error_flag}\n"
        
        os.makedirs("data", exist_ok=True)
        with open("data/api_traffic.log", "a") as f:
            f.write(log_line)
            
        return response