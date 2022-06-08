import supervisely as sly

server_address = "my-private-supervisely-instance.com"
api_token = "my API token"
project_id = 11252

api = sly.Api(server_address, api_token)

new_readme = """
# title

123

## subtitle

321
"""

api.project.edit_info(project_id, readme=new_readme)


api.project.edit_info(
    project_id,
    name="my new name",
    description="my new description",
    readme=new_readme,
    custom_data={"a": "abc", "b": 123},
)
