import supervisely as sly
from dotenv import load_dotenv
import os
from time import sleep
from tqdm import tqdm

load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()
os.environ["ENV"] = "production"


class MyTqdm(sly.Progress, tqdm):
    def __init__(self, *args, **kwargs):

        self.temp_current = 0

        super().__init__(*args, **kwargs)

    def write(self, count):
        if sly.is_development():
            self.temp_current += count
            if self.temp_current != self.total:
                pass
            else:
                for _ in tqdm(range(self.total)):
                    sleep(0.01)
        else:
            super().iters_done_report(count)


data_len = 120
data_list = [i for i in range(data_len)]


def test_progress(x, progress):

    batch_size = 30
    for i in sly.batched(seq=x, batch_size=batch_size):
        progress.write(batch_size)


progress_tqdm = MyTqdm(message="Uploading data:", total_cnt=data_len)
test_progress(data_list, progress_tqdm)
