import supervisely as sly

server_address = ""
token = ""

api = sly.Api(server_address=server_address, token=token)

team_id = 8
workspace_id = 349
project_id = 13933
dataset_id = 51608

image_id = 8498108
image_ids = [8498105, 8498107]

image_name = "doge.png"
images_names = [
    "40_doge.png",
    "40_test-s3-ny_S3_links-ny_000163-0.jpg",
    "40_test-s3-ny_S3_links-ny_000163-1.jpg",
    "40_test-s3-ny_S3_links-ny_000248-1.jpg",
    "40_cat-aaa.jpeg",
]
image_link = "s3://remote-img-test/doge.png"
images_links = [
    "s3://remote-img-test/doge.png",
    "s3://remote-img-test/test-s3-ny_S3_links-ny_000163-0.jpg",
    "s3://remote-img-test/test-s3-ny_S3_links-ny_000163-1.jpg",
    "s3://remote-img-test/test-s3-ny_S3_links-ny_000248-1.jpg",
    "s3://remote-img-test/cat-aaa.jpeg",
]

image_info_by_id = api.image.get_info_by_id(id=image_id, force_metadata_for_links=False)
print("---------------------------")
print("------ GET INFO BY ID -----")
print("---------------------------")
print(image_info_by_id)

image_info_by_name = api.image.get_info_by_name(
    dataset_id=dataset_id, name=image_name, force_metadata_for_links=False
)
print("---------------------------")
print("---- GET INFO BY NAME -----")
print("---------------------------")
print(image_info_by_name)

images_infos_by_batch_ids = api.image.get_info_by_id_batch(
    ids=image_ids, force_metadata_for_links=False
)
print("---------------------------")
print("-- GET INFO BY BATCH IDS --")
print("---------------------------")
print(images_infos_by_batch_ids)

images_infos = api.image.get_list(dataset_id=dataset_id, force_metadata_for_links=False)
print("---------------------------")
print("-------- GET LIST ---------")
print("---------------------------")
print(images_infos)

image_info_upload_link = api.image.upload_link(
    dataset_id=dataset_id,
    name=f"44_{image_name}",
    link=image_link,
    force_metadata_for_links=False,
)
print("---------------------------")
print("-------- BY LINK ----------")
print("---------------------------")
print(image_info_upload_link)

images_infos_upload_links = api.image.upload_links(
    dataset_id=dataset_id,
    names=images_names,
    links=images_links,
    force_metadata_for_links=False,
)
print("---------------------------")
print("-------- BY LINKS ---------")
print("---------------------------")
print(images_infos_upload_links)
