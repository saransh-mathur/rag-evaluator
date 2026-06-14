# Async Python: Avoid Blocking the Event Loop

When writing asynchronous Python code with `asyncio`, never call blocking functions like `time.sleep()` inside an `async def` coroutine. Blocking calls freeze the entire event loop and prevent other coroutines from running.

## Use asyncio.sleep instead

Replace synchronous sleep with the async equivalent:

```python
import asyncio

async def worker():
    # Wrong: blocks the event loop
    # time.sleep(1)

    # Correct: yields control back to the event loop
    await asyncio.sleep(1)
```

`await asyncio.sleep(seconds)` schedules a non-blocking delay. While waiting, the event loop can run other tasks.

## Why blocking matters

The event loop multiplexes many coroutines on a single thread. A blocking call such as `time.sleep()` or heavy synchronous I/O stops scheduling until it returns. Under load this causes latency spikes and timeouts.

## Practical guidance

- Use `asyncio.sleep` for delays in async code.
- Offload CPU-bound or legacy blocking I/O to `asyncio.to_thread()` or an executor.
- Prefer async-native libraries (`aiohttp`, `asyncpg`) over blocking clients wrapped in threads when possible.
