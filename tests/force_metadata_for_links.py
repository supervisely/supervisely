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

# Images names
# [
#     'doge.png',
#     'test-s3-ny_S3_links-ny_000163-0.jpg',
#     'test-s3-ny_S3_links-ny_000163-1.jpg',
#     'test-s3-ny_S3_links-ny_000248-1.jpg', 
#     'cat-aaa.jpeg'
#  ]


image_info_by_id  = api.image.get_info_by_id(id=image_id, force_metadata_for_links=False)
print("---------------------------")
print("------ GET INFO BY ID -----")
print("---------------------------")
print(image_info_by_id)

image_info_by_name = api.image.get_info_by_name(dataset_id=dataset_id, name=image_name, force_metadata_for_links=False)
print("---------------------------")
print("---- GET INFO BY NAME -----")
print("---------------------------")
print(image_info_by_name)

images_infos_by_batch_ids = api.image.get_info_by_id_batch(ids=image_ids, force_metadata_for_links=False)
print("---------------------------")
print("-- GET INFO BY BATCH IDS --")
print("---------------------------")
print(images_infos_by_batch_ids)

images_infos = api.image.get_list(dataset_id=dataset_id, force_metadata_for_links=False)
print("---------------------------")
print("-------- GET LIST ---------")
print("---------------------------")
print(images_infos)
