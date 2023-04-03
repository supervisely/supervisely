import supervisely as sly
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()


class MyTqdm(sly.Progress, tqdm):
    def __init__(self, message, total_cnt):

        self.message = message
        self.total = total_cnt
        super().__init__(message=message, total_cnt=total_cnt)

    def write(self, count):
        super().iters_done_report(count)


data_len = 120
data_list = [i for i in range(data_len)]


def test_progress(x, progress):

    batch_size = 30
    if type(progress) == sly.Progress:
        for i in sly.batched(seq=x, batch_size=batch_size):
            progress.iters_done_report(batch_size)
    else:
        for i in sly.batched(seq=x, batch_size=batch_size):
            progress.write(batch_size)


progress = sly.Progress("Uploading data:", total_cnt=data_len)
test_progress(data_list, progress)


progress_tqdm = MyTqdm(message="Uploading data:", total_cnt=data_len)
test_progress(data_list, progress_tqdm)
