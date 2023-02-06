import os
from dotenv import load_dotenv

import supervisely as sly


# load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

# team_id = sly.env.team_id()
team_id = 435
PATH = "/sly-app-data"


files_list = api.file.list(team_id, PATH)

print('Total {} files in "{}" directory.'.format(len(files_list), PATH))
print(f"First file in list: {files_list[0]}")
# Output: 
# Total 29 files in "/sly-app-data" directory.
# First file in list: {'id': 772315, 'isDir': True, 'meta': {'size': 8605}, 'path': '/sly-app-data/ma...}


files_list_2 = api.file.list2(team_id, PATH)

print('Total {} files in "{}" directory.'.format(len(files_list_2), PATH))
print(f"First FileInfo object in list: {files_list_2[0]}")
# Output: 
# Total 29 files in "/sly-app-data" directory.
# First FileInfo object in list: FileInfo(team_id=435, id=18421, user_id=330, name='507_002.tar.gz', hash='+0nrNoDjBxxJA...


listdir = api.file.listdir(team_id, PATH)

print(listdir)
# Output: [
#     '/sly-app-data/object-tags-editor-files',
#     '/sly-app-data/mark-segments-on-synced-videos-2-files'
# ]


listdir_s = api.file.listdir(team_id, f"{PATH}/")

print(listdir_s)
# before update SDK
# Output: [] # return [] if path ends with "/"

# after update SDK pr#382 
# Output: [
#     '/sly-app-data/object-tags-editor-files',
#     '/sly-app-data/mark-segments-on-synced-videos-2-files'
# ]


###############################################################################
# https://github.com/supervisely/supervisely/pull/390
# added new parameter recursive in list, list2, listdir methods
# fixed path separator in listdir method


list_not_recursively = api.file.list(team_id, PATH, recursive=False) # default recursive=True

print(list_not_recursively)
# Output: [
#     {'id': 772315, 'isDir': True, 'meta': {'size': 8605}, 'path': '/sly-app-data/ma...},
#     {'id': 847993, 'isDir': True, 'meta': {'size': 277}, 'path': '/sly-app-data/ob...}
# ]


list2_not_recursively = api.file.list2(team_id, PATH, recursive=False) # default recursive=True

print(list2_not_recursively)
# Output: [
#     FileInfo(team_id=435, id=18421, user_id=330, name='50t_video_002.tar.gz', hash='+0nrNoDjBxxJA...
#     FileInfo(team_id=435, id=18431, user_id=330, name='50t_video_002.tar.gz', hash='+0nrNoDjBxxJA...
# ]


listdir_recursively = api.file.listdir(team_id, PATH, recursive=True)

print('{} paths has been collected from "{}" directory.'.format(len(listdir_recursively), PATH))
print(f"First path in list: {listdir_recursively[0]}")
# Output: 
# 29 paths has been collected from "/sly-app-data" directory.
# "First path in list: '/sly-app-data/mark-segments-on-synced-videos-2-files/project-15918/dataset-54074/Info.json'


listdir_recursively_s = api.file.listdir(team_id, f"{PATH}/", recursive=True)

print('{} paths has been collected from "{}" directory.'.format(len(listdir_recursively_s), PATH))
print(f"First path in list: {listdir_recursively_s[0]}")
# Output: 
# 29 paths has been collected from "/sly-app-data" directory.
# "First path in list: '/sly-app-data/mark-segments-on-synced-videos-2-files/project-15918/dataset-54074/Info.json'
