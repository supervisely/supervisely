import asyncio
import concurrent.futures


def run_sync(coroutine):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(lambda coroutine_to_exec: asyncio.run(coroutine_to_exec), coroutine).result()
    else:
        result = asyncio.run(coroutine)
    return result
