import asyncio


def run_sync(coroutine):
    try:
        return asyncio.run_coroutine_threadsafe(
            coro=asyncio.to_thread(coroutine), loop=asyncio.get_event_loop()
        ).result()
    except RuntimeError as ex:
        print(repr(ex))
        return None
