import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()


src_path = "/reports/objects_stats/admin/"
# src_path = "agent://42/test-import"
files = api.file.list2(team_id=7, path=src_path)
for file_path in files:
    print(file_path)

# team files:
# data = {
#     "id": 8,
#     "userId": 1,
#     "path": "/reports/objects_stats/admin/ecosystem/lemons_annotated.lnk",
#     "storagePath": "..../teams_storage/7/7/5/kk/iZrRbQsYPWRZL2zfYfXGUiPihig7bIMksGFZsuY2Qczep49VXpAruXEq1KwmFFSkHpS9uiR7GDspvYSaQ0DEzTr4hlkEp4gdmOKaKjUTd4T59gOV18MWOuuJ9DKu.txt",
#     "meta": {"ext": "lnk", "mime": "text/plain", "size": 45},
#     "createdAt": "2020-10-26T09:20:18.967Z",
#     "updatedAt": "2020-10-26T09:20:18.967Z",
#     "hash": "skf1eAROD7JuPD1z4woSPmo0E+jiXcfk0tywXIwLlnk=",
#     "fullStorageUrl": "https:///.../teams_storage/7/7/5/kk/iZrRbQsYPWRZL2zfYfXGUiPihig7bIMksGFZsuY2Qczep49VXpAruXEq1KwmFFSkHpS9uiR7GDspvYSaQ0DEzTr4hlkEp4gdmOKaKjUTd4T59gOV18MWOuuJ9DKu.txt",
#     "teamId": 7,
#     "name": "lemons_annotated.lnk",
# }

# agent team files
