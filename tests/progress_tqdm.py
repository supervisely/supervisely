from __future__ import annotations

import supervisely as sly
from dotenv import load_dotenv
import os
from time import sleep
from tqdm import tqdm
from typing import Optional
from supervisely.sly_logger import logger


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()
# os.environ["ENV"] = "production"


class MyTqdm(tqdm):
    def __init__(
        self,
        message,
        total_cnt,
        ext_logger: Optional[logger] = None,
        is_size: Optional[bool] = False,
        need_info_log: Optional[bool] = False,
        min_report_percent: Optional[int] = 1,
        *args,
        **kwargs,
    ):

        self.message = message
        self.total = total_cnt
        self.ext_logger = ext_logger
        self.is_size = is_size
        self.need_info_log = need_info_log
        self.min_report_percent = min_report_percent

        if sly.is_production():
            self.progress = sly.Progress(
                message=self.message,
                total_cnt=self.total,
                ext_logger=self.ext_logger,
                is_size=self.is_size,
                need_info_log=self.need_info_log,
                min_report_percent=self.min_report_percent,
            )

        super().__init__(total=total_cnt, desc=message, *args, **kwargs)

    def write(self, count):
        if sly.is_development():
            for _ in range(count):
                super().update(1)
                sleep(0.1)
        else:
            self.progress.iters_done_report(count)


data_len = 145
data_list = [i for i in range(data_len)]


def test_progress(x, progress):

    batch_size = 30
    for batch in sly.batched(seq=x, batch_size=batch_size):
        progress.write(len(batch))


progress_tqdm = MyTqdm(message="Uploading data", total_cnt=data_len)
test_progress(data_list, progress_tqdm)


# size = api.file.get_directory_size(439, "/test.tar")
# progress_tqdm_file = MyTqdm("File downloaded: ", total_cnt=size, is_size=True)
# api.file.download(
#     team_id=439,
#     remote_path="/test.tar",
#     local_save_path="/home/alex/Downloads/test.tar",
#     progress_cb=progress_tqdm_file.write,
# )
