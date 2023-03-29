import traceback

import supervisely as sly

def get_project_name_run(project_id:int) -> bool:

    if sly.is_development():
        sly.Api.from_env_file()
       
    api = sly.Api.from_env()

    try:
        project_info = api.project.get_info_by_id(project_id)
        if project_info is None:
            print(f"Project with id {project_id} is either archived, doesn't exist or you don't have enough permissions to access it")
            return False
        
        print(project_info.name)
        return True
    
    except:
        traceback.print_exc()
        return False
