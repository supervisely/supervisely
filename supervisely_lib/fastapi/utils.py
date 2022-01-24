import os
import signal
import psutil
# from fastapi import FastAPI


def graceful_shutdown(): #app: FastAPI):
    # https://github.com/tiangolo/fastapi/issues/1509
    current_process = psutil.Process(os.getpid())
    current_process.send_signal(signal.SIGINT) # emit ctrl + c