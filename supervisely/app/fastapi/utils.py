import asyncio
import concurrent.futures


def run_sync(coroutine):
    """
    Runs an asynchronous coroutine in a separate thread and waits for its result.
    It is useful for running async functions in a synchronous
    environment.

    This method creates a new thread using ThreadPoolExecutor and executes the coroutine
    inside a new event loop.

    ⚠️ Note: This function creates a new event loop every time it is called,
    which can cause issues when using objects tied to a specific loop (e.g., asyncio.Semaphore).

    :param coroutine: coroutine to run
    :return: result of coroutine
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                lambda coroutine_to_exec: asyncio.run(coroutine_to_exec), coroutine
            ).result()
        return result
    except RuntimeError as ex:
        print(repr(ex))
        return None
