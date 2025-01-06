import asyncio
import os

import supervisely as sly

api = sly.Api.from_env()
TEAM_ID = int(os.environ["TEAM_ID"])
# test_dir = "/test_resumable_api"
test_dir = "/test_images_download"
# test_file = "cdandcinuav.tar"
test_file = "yolov8s-seg.pt"
test_path = os.path.join(test_dir, test_file)
dest_dir = "/test_resumable_api"
# dest_dir = "google://sly-dev-test/resumable-test"  # not implemented
# dest_dir = "s3://bucket-test-export/resumable-test"
# dest_dir = "fs://test/resumable-test" # not implemented
# dest_dir = "azure://supervisely-test/resumable-test"

sha256 = sly.fs.get_file_hash(test_path)
crc32 = sly.fs.get_file_hash(test_path, "crc32")
blake3 = sly.fs.get_file_hash(test_path, "blake3")
size = sly.fs.get_file_size(test_path)
tqdm = sly.tqdm_sly(desc="Uploading file", total=size, unit="B", unit_scale=True)

loop = sly.utils.get_or_create_event_loop()
coro = api.file.upload_resumable_async(
    team_id=TEAM_ID,
    src=test_path,
    dst=os.path.join(dest_dir, test_file),
    progress_cb=tqdm,
    autocomplete=True,
)
if loop.is_running():
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    result = future.result()
else:
    result = loop.run_until_complete(coro)

