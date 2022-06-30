import asyncio
import concurrent.futures

from supervisely import logger


def run_sync(coroutine):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(lambda coroutine_to_exec: asyncio.run(coroutine_to_exec), coroutine).result()
    return result
