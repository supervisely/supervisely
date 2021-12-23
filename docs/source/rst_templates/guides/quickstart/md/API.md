
## Initialize API object

### Initialize

Direct initialization:

<span style="color: green;">Input:</span>

    import supervisely_lib as sly
    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)

Initialize from environment variables:

Create two environment variables - SERVER_ADDRESS and API_TOKEN:

<span style="color: green;">Input:</span>

    print(os.environ["SERVER_ADDRESS"])
    print(os.environ["API_TOKEN"])

<span style="color: blue;">Output:</span>

    https://app.supervise.ly
    P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww

Use from_env() function to initialize api:

<span style="color: green;">Input:</span>

    api = sly.Api.from_env()


### Check connection with api

To check that connection with api is successful get list of teams in your account:

<span style="color: green;">Input:</span>

    team_list = api.team.get_list()
    print(team_list)

<span style="color: blue;">Output:</span>

    [TeamInfo(id=16087, name='alexxx', description='', role='admin', created_at='2019-08-02T09:31:05.860Z', updated_at='2019-08-02T09:31:05.860Z')]


### Error handling

If the connection to the site you specified is not established raise RetryError:

<span style="color: green;">Input:</span>

    try:
        team_list = api.team.get_list()
        print(team_list)
    except requests.exceptions.RetryError as error:
        print(error)

<span style="color: blue;">Output:</span>

    Retry limit exceeded

If the token you specified is not correct or does not exist raise HTTPError:

<span style="color: blue;">Output:</span>

    401 Client Error: Unauthorized for url: https://app.supervise.ly/public/api/v3/teams.list ({"error":"Unauthorized","details":null})


### Configure api

<span style="color: green;">Input:</span>

    api = sly.Api(address, token, retry_count=5, retry_sleep_sec=3)

### Progress and function wrapper

Progress use for conveniently monitoring the operation of modules and displaying statistics on data processing:

<span style="color: green;">Input:</span>

    progress = sly.Progress("Example progress", api.project.get_images_count(103287))
    for src_dataset in api.dataset.get_list(103287):
            images = api.image.get_list(src_dataset.id)
            for batch in sly.batched(images):
                progress.iters_done_report(len(batch))

<span style="color: blue;">Output:</span>

    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 0, "total": 351, "timestamp": "2021-02-14T14:39:59.177Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 50, "total": 351, "timestamp": "2021-02-14T14:39:59.831Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 100, "total": 351, "timestamp": "2021-02-14T14:39:59.831Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 150, "total": 351, "timestamp": "2021-02-14T14:39:59.832Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 200, "total": 351, "timestamp": "2021-02-14T14:39:59.832Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 250, "total": 351, "timestamp": "2021-02-14T14:39:59.832Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 300, "total": 351, "timestamp": "2021-02-14T14:39:59.832Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 350, "total": 351, "timestamp": "2021-02-14T14:39:59.832Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 351, "total": 351, "timestamp": "2021-02-14T14:39:59.832Z", "level": "info"}

Using is_size flag:

<span style="color: green;">Input:</span>

    progress = sly.Progress("Example progress", 0, is_size=True)
    for src_dataset in api.dataset.get_list(102344):
            images = api.image.get_list(src_dataset.id)
            for batch in sly.batched(images, batch_size=200):
                progress.iters_done_report(len(batch))

<span style="color: blue;">Output:</span>

    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 0, "total": 0, "current_label": "0.0 B", "total_label": "0.0 B", "timestamp": "2021-02-14T14:50:31.175Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 200, "total": 200, "current_label": "200.0 B", "total_label": "200.0 B", "timestamp": "2021-02-14T14:50:33.140Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 400, "total": 400, "current_label": "400.0 B", "total_label": "400.0 B", "timestamp": "2021-02-14T14:50:33.141Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 600, "total": 600, "current_label": "600.0 B", "total_label": "600.0 B", "timestamp": "2021-02-14T14:50:33.141Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 800, "total": 800, "current_label": "800.0 B", "total_label": "800.0 B", "timestamp": "2021-02-14T14:50:33.141Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 1000, "total": 1000, "current_label": "1000.0 B", "total_label": "1000.0 B", "timestamp": "2021-02-14T14:50:33.141Z", "level": "info"}
    {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Example progress", "current": 1200, "total": 1200, "current_label": "1.2 KiB", "total_label": "1.2 KiB", "timestamp": "2021-02-14T14:50:33.141Z", "level": "info"}

Logging critical error in function if it occured:

<span style="color: green;">Input:</span>

    def example():
        return 5 / 0
    
    sly.main_wrapper('wrap_ex', example)

<span style="color: blue;">Output:</span>

    {"message": "Unexpected exception in main.", "main_name": "func_name", "event_type": "EventType.TASK_CRASHED", "exc_str": "division by zero", "timestamp": "2021-02-14T15:11:56.399Z", "level": "fatal", "stack": ["Traceback (most recent call last):", "  File \"/home/sdk_docs_examples/venv/lib/python3.8/site-packages/supervisely_lib/function_wrapper.py\", line 15, in main_wrapper", "    main_func(*args, **kwargs)", "  File \"/home/sdk_docs_examples/001_api.py\", line 43, in example", "    return 5 / 0", "ZeroDivisionError: division by zero"]}

## Teams and workspaces

### Teams

Get list of teams in your account:

<span style="color: green;">Input:</span>

    team_list = api.team.get_list()
    for team in team_list:
        print(team)

<span style="color: blue;">Output:</span>

    TeamInfo(id=16087, name='alexxx', description='', role='admin', created_at='2019-08-02T09:31:05.860Z', updated_at='2019-08-02T09:31:05.860Z')
    TeamInfo(id=41293, name='test_team', description='', role='admin', created_at='2021-01-22T12:19:54.371Z', updated_at='2021-01-22T12:19:54.371Z')

Get information about team by it ID:

<span style="color: green;">Input:</span>

    team_info = api.team.get_info_by_id(16087)
    print(team_info)

<span style="color: blue;">Output:</span>

    [TeamInfo(id=16087, name='alexxx', description='', role='admin', created_at='2019-08-02T09:31:05.860Z']

If team with ID is either archived, doesn't exist or you don't have enough permissions to access it, warn message will be generated. The request will return a value None.

Create team with given name in your account:

<span style="color: green;">Input:</span>

    new_team = api.team.create('my_new_team')
    print(new_team)

<span style="color: blue;">Output:</span>

    TeamInfo(id=41898, name='my_new_team', description='', role='admin', created_at='2021-02-06T09:42:39.375Z', updated_at='2021-02-06T09:42:39.375Z')

If team with given name already exist raise HTTPError:

<span style="color: blue;">Output:</span>

    400 Client Error: Bad Request for url: https://app.supervise.ly/public/api/v3/teams.add ({"error":"groups already exists","details":{"type":"NONUNIQUE","errors":[{"name":"my_new_team","id":1036,"message":"Team with name \"my_new_team\" already exists"}]}})

To avoid error, use 'change_name_if_conflict' flag:

<span style="color: green;">Input:</span>

    new_team = api.team.create('my_new_team', change_name_if_conflict=True)
    print(new_team)

<span style="color: blue;">Output:</span>

    TeamInfo(id=41899, name='my_new_team_001', description='', role='admin', created_at='2021-02-06T09:44:25.268Z', updated_at='2021-02-06T09:44:25.268Z')
    
### Workspaces

Get list of all the workspaces in the selected team:

<span style="color: green;">Input:</span>

    workspaces = api.workspace.get_list(16087)
    for workspace in workspaces:
        print(workspace)

<span style="color: blue;">Output:</span>

    WorkspaceInfo(id=17190, name='First Workspace', description='', team_id=16087, created_at='2019-08-02T09:31:05.860Z', updated_at='2019-08-02T09:31:05.860Z')
    WorkspaceInfo(id=23821, name='my_super_workspace', description='super workspace description', team_id=16087, created_at='2019-12-19T11:41:55.494Z', updated_at='2019-12-19T11:41:55.494Z')
    WorkspaceInfo(id=23855, name='api_inference_tutorial', description='', team_id=16087, created_at='2019-12-20T10:21:13.265Z', updated_at='2019-12-20T10:21:13.265Z')

Get information about workspace by it ID:

<span style="color: green;">Input:</span>

    workspace = api.workspace.get_info_by_id(17190)
    print(workspace)

<span style="color: blue;">Output:</span>

    WorkspaceInfo(id=17190, name='First Workspace', description='', team_id=16087, created_at='2019-08-02T09:31:05.860Z', updated_at='2019-08-02T09:31:05.860Z')

If workspace with ID is either archived, doesn't exist or you don't have enough permissions to access it, warn message will be generated. The request will return a value None.

Create workspace with given name in team with given id:

<span style="color: green;">Input:</span>

    new_workspace = api.workspace.create(16087, 'new_workspace')
    print(new_workspace)

<span style="color: blue;">Output:</span>

    WorkspaceInfo(id=48467, name='new_workspace', description='', team_id=16087, created_at='2021-02-06T09:56:35.690Z', updated_at='2021-02-06T09:56:35.690Z')

If workspace with given name already exist raise HTTPError:

<span style="color: blue;">Output:</span>

    400 Client Error: Bad Request for url: https://app.supervise.ly/public/api/v3/workspaces.add ({"error":"workspaces already exists","details":{"type":"NONUNIQUE","errors":[{"name":"new_workspace","id":48467,"message":"Workspace with name \"new_workspace\" already exists"}]}})

To avoid error, use 'change_name_if_conflict' flag:

<span style="color: green;">Input:</span>

    new_workspace = api.workspace.create(16087, 'new_workspace', change_name_if_conflict=True)
    print(new_workspace)

<span style="color: blue;">Output:</span>

    WorkspaceInfo(id=48468, name='new_workspace_001', description='', team_id=16087, created_at='2021-02-06T09:59:17.123Z', updated_at='2021-02-06T09:59:17.123Z')
