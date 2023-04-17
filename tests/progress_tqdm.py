from __future__ import annotations

import supervisely as sly
from dotenv import load_dotenv
import os, math
from time import sleep
from tqdm import tqdm
from typing import Optional
from supervisely.sly_logger import logger

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")
api = sly.Api()
# os.environ["ENV"] = "production"

print("___________START_________-")

TF_PATH = "/cardio.tar"
LOC_PATH = "prog_test.tar"
TEAM_ID = sly.env.team_id()


class MyTqdm(tqdm, sly.Progress):
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

        self.current = 0
        self.is_total_unknown = total_cnt == 0

        self.total_label = ""
        self.current_label = ""
        self.reported_cnt = 0
        self.logger = logger if ext_logger is None else ext_logger

        self.report_every = max(1, math.ceil(total_cnt / 100 * min_report_percent))

        if sly.is_development():
            super().__init__(
                total=total_cnt,
                desc=message,
                *args,
                **kwargs,
            )
        else:
            super().__init__(
                disable=True,
                total=total_cnt,
                *args,
                **kwargs,
            )

    def update(self, count):
        if sly.is_development():
            super().update(count)
        else:
            super().iters_done_report(count)


# data_len = 145
# data_list = list(range(data_len))


# def test_progress(x, progress):
#     batch_size = 30
#     for batch in sly.batched(seq=x, batch_size=batch_size):
#         progress.update(len(batch))


p = MyTqdm(
    message="Downloading data with MyTqdm",
    total_cnt=api.file.get_directory_size(TEAM_ID, TF_PATH),
    unit="B",
    unit_scale=True,
)

# p = tqdm(
#     desc="Downloading data with original tqdm",
#     total=api.file.get_directory_size(TEAM_ID, TF_PATH),
#     unit="B",
#     unit_scale=True,
# )


# progress_tqdm
# test_progress(data_list, progress_tqdm)
# progress_tqdm.close()

api.file.download(TEAM_ID, TF_PATH, LOC_PATH, progress_cb=p.update)

print(os.getcwd())
print(os.listdir(os.getcwd()))
# size = api.file.get_directory_size(439, "/test.tar")
# progress_tqdm_file = MyTqdm("File downloaded: ", total_cnt=size, is_size=True)
# api.file.download(
#     team_id=439,
#     remote_path="/test.tar",
#     local_save_path="/home/alex/Downloads/test.tar",
#     progress_cb=progress_tqdm_file.update,
# )
# progress_tqdm_file.close()
