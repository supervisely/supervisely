import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

counter = 0
# batch_size = 100
batch_size = 20000  # use large batch size if you have huge datasets
for batch in api.image.get_list_generator(dataset_id=1788, batch_size=batch_size):
    image_ids = [info.id for info in batch]
    image_names = [info.name for info in batch]

    counter += len(image_ids)
    print(f"Already processed: {counter} images")
