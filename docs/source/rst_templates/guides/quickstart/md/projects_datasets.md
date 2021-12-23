
## Projects and datasets in API

### Projects in API

Get list of all the projects in the selected workspace:

    projects = api.project.get_list(23821)
    for project in projects:
        print(project)

    ProjectInfo(id=53939, name='lemons_annotated_clone', description='', size='861069', readme='', workspace_id=23821, images_count=None, items_count=None, datasets_count=None, created_at='2019-12-19T12:06:59.435Z', updated_at='2020-08-31T12:01:42.943Z', type='images', reference_image_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/t/t5/VMCn7mwooVoNAokqxvpNPVRcUW52B0zPp2CzSz1tQP6l0H5xAO8zX9iuT6CmAggtnjVJ0tjZ9taJ5ChiC9rvmz8plmIOSViFBIePEyslSFYmFpzWf7Rf4rN8iIXx.jpg')
    ProjectInfo(id=91678, name='geometry_diff', description='', size='861069', readme='', workspace_id=23821, images_count=None, items_count=None, datasets_count=None, created_at='2020-08-18T12:36:19.015Z', updated_at='2020-08-19T09:13:06.200Z', type='images', reference_image_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/t/t5/VMCn7mwooVoNAokqxvpNPVRcUW52B0zPp2CzSz1tQP6l0H5xAO8zX9iuT6CmAggtnjVJ0tjZ9taJ5ChiC9rvmz8plmIOSViFBIePEyslSFYmFpzWf7Rf4rN8iIXx.jpg')
    ProjectInfo(id=92040, name='pc', description='', size='1940352', readme='', workspace_id=23821, images_count=None, items_count=None, datasets_count=None, created_at='2020-08-23T08:50:13.228Z', updated_at='2020-08-23T09:09:25.944Z', type='point_clouds', reference_image_url=None)

Get information about project by it ID:

    project = api.project.get_info_by_id(53939)
    print(project)

    ProjectInfo(id=53939, name='lemons_annotated_clone', description='', size='861069', readme='', workspace_id=23821, images_count=6, items_count=6, datasets_count=1, created_at='2019-12-19T12:06:59.435Z', updated_at='2020-08-31T12:01:42.943Z', type='images')

If project with ID is either archived, doesn't exist or you don't have enough permissions to access it, warn message will be generated. The request will return a value None.

Get information about project by it name in the selected workspace:

    project = api.project.get_info_by_name(23821, 'lemons_annotated')
    print(project)

    ProjectInfo(id=53939, name='lemons_annotated_clone', description='', size='861069', readme='', workspace_id=23821, images_count=None, items_count=None, datasets_count=None, created_at='2019-12-19T12:06:59.435Z', updated_at='2020-08-31T12:01:42.943Z', type='images', reference_image_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/t/t5/VMCn7mwooVoNAokqxvpNPVRcUW52B0zPp2CzSz1tQP6l0H5xAO8zX9iuT6CmAggtnjVJ0tjZ9taJ5ChiC9rvmz8plmIOSViFBIePEyslSFYmFpzWf7Rf4rN8iIXx.jpg')

Create project with given name in workspace with given id:

    new_project = api.project.create(23821, 'new_project_name')
    print(new_project)

    ProjectInfo(id=115715, name='new_project_name', description='', size='0', readme='', workspace_id=23821, images_count=None, items_count=None, datasets_count=None, created_at='2021-02-06T10:53:56.894Z', updated_at='2021-02-06T10:53:56.894Z', type='images', reference_image_url=None)

If project with given name already exist raise HTTPError:

    400 Client Error: Bad Request for url: https://app.supervise.ly/public/api/v3/projects.add ({"error":"project already exists","details":{"type":"NONUNIQUE","errors":[{"name":"new_project","id":48467,"message":"Project with name \"new_project\" already exists"}]}})

To avoid error, use 'change_name_if_conflict' flag:

    new_project = api.project.create(23821, 'new_project', change_name_if_conflict=True)
    print(new_project)

    ProjectInfo(id=115716, name='new_project_001', description='', size='0', readme='', workspace_id=23821, images_count=None, items_count=None, datasets_count=None, created_at='2021-02-06T10:55:57.130Z', updated_at='2021-02-06T10:55:57.130Z', type='images', reference_image_url=None)

Get project meta by project ID:

    projectmeta = api.project.get_meta(53939)
    print(projectmeta)

    {'classes': [{'id': 1168245, 'title': 'kiwi', 'shape': 'bitmap', 'hotkey': '', 'color': '#FF0000'}, {'id': 1168244, 'title': 'lemon', 'shape': 'bitmap', 'hotkey': '', 'color': '#51C6AA'}], 'tags': [], 'projectType': 'images'}

Update project meta in given project:

    new_meta = {'classes': [{'id': 1168245, 'title': 'apple', 'shape': 'bitmap', 'hotkey': '', 'color': '#00FF00'}], 'tags': [], 'projectType': 'images'}
    api.project.update_meta(53939, new_meta)
    new_projectmeta = api.project.get_meta(53939)
    print(new_projectmeta)

    {'classes': [{'id': 2788917, 'title': 'apple', 'shape': 'bitmap', 'hotkey': '', 'color': '#00FF00'}], 'tags': [], 'projectType': 'images'}

Get number of datasets in given project:

    print(api.project.get_datasets_count(53939))

    2

Get number of images in given project:

    print(api.project.get_images_count(53939))

    6

Add metadata from given project to given destination project:

    print(api.project.get_meta(98674))
    print(api.project.get_meta(53939))
    new_dst_meta_json = api.project.merge_metas(98674, 53939)
    print(new_dst_meta_json)

    {'classes': [{'id': 2169600, 'title': 'kiwi', 'shape': 'bitmap', 'hotkey': '', 'color': '#FF0000'}, {'id': 2169599, 'title': 'lemon', 'shape': 'bitmap', 'hotkey': '', 'color': '#51C6AA'}], 'tags': [], 'projectType': 'images'}
    {'classes': [{'id': 2788917, 'title': 'apple', 'shape': 'bitmap', 'hotkey': '', 'color': '#00FF00'}], 'tags': [], 'projectType': 'images'}
    {'classes': [{'title': 'apple', 'shape': 'bitmap', 'color': '#00FF00', 'geometry_config': {}, 'id': 2788917, 'hotkey': ''}, {'title': 'kiwi', 'shape': 'bitmap', 'color': '#FF0000', 'geometry_config': {}, 'id': 2169600, 'hotkey': ''}, {'title': 'lemon', 'shape': 'bitmap', 'color': '#51C6AA', 'geometry_config': {}, 'id': 2169599, 'hotkey': ''}], 'tags': [], 'projectType': 'images'}

Get url for project with given ID:

    print(api.project.url(53939))

    https://app.supervise.ly/projects/53939/datasets

### Datasets in API

Get list of all the datasets in the selected project:

    datasets = api.dataset.get_list(102344)
    for dataset in datasets:
        print(dataset)

    DatasetInfo(id=393776, name='val', description='', size='161555379', project_id=102344, images_count=1449, items_count=1449, created_at='2020-10-20T16:15:34.729Z', updated_at='2020-10-20T16:15:37.796Z', reference_image_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/x/UB/nx4ygRLhY1fqjDWlq3TJWUkq2PTD3wBincrC0Rx436rJkfHTJ0tKRrDG119u0OQgVnakzd52Rv7ZX4P8niocCc8V74eIKKBthrBnrXAAKuhI9xGFkujXElSrTn8G.jpg')
    DatasetInfo(id=393777, name='train', description='', size='161296236', project_id=102344, images_count=1464, items_count=1464, created_at='2020-10-20T16:15:52.710Z', updated_at='2020-10-20T16:15:55.719Z', reference_image_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/h/N/fO/A0KjO2S95f9yGKgWGRUxPc9cPcrDPpFPK6MoGiQfBwEylPe4IM8NOgLSnB1Sr87kQ60E3AAzsq7PvPkyjeSu5cbd7iAM1j1C5os7ET5Vkl4LL4ugNFscv6Z85QKX.jpg')
    DatasetInfo(id=393778, name='trainval', description='', size='322851615', project_id=102344, images_count=2913, items_count=2913, created_at='2020-10-20T16:16:08.586Z', updated_at='2020-10-20T16:16:11.659Z', reference_image_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/h/N/fO/A0KjO2S95f9yGKgWGRUxPc9cPcrDPpFPK6MoGiQfBwEylPe4IM8NOgLSnB1Sr87kQ60E3AAzsq7PvPkyjeSu5cbd7iAM1j1C5os7ET5Vkl4LL4ugNFscv6Z85QKX.jpg')

Get information about dataset by it ID:

    dataset = api.dataset.get_info_by_id(393776)
    print(dataset)

    DatasetInfo(id=393776, name='val', description='', size='161555379', project_id=102344, images_count=1449, items_count=1449, created_at='2020-10-20T16:15:34.729Z', updated_at='2020-10-20T16:15:37.796Z', reference_image_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/x/UB/nx4ygRLhY1fqjDWlq3TJWUkq2PTD3wBincrC0Rx436rJkfHTJ0tKRrDG119u0OQgVnakzd52Rv7ZX4P8niocCc8V74eIKKBthrBnrXAAKuhI9xGFkujXElSrTn8G.jpg')

If dataset with ID is either archived, doesn't exist or you don't have enough permissions to access it, warn message will be generated. The request will return a value None.

Create dataset with given name in project with given id:

    new_dataset = api.dataset.create(53939, 'new_dataset')
    print(new_dataset)

    DatasetInfo(id=441021, name='new_dataset', description='', size='0', project_id=53939, images_count=0, items_count=0, created_at='2021-02-06T11:32:47.220Z', updated_at='2021-02-06T11:32:47.220Z', reference_image_url=None)

If dataset with given name already exist in project raise HTTPError:

    400 Client Error: Bad Request for url: https://app.supervise.ly/public/api/v3/datasets.add ({"error":"dataset already exists","details":{"type":"NONUNIQUE","errors":[{"name":"new_dataset","id":48467,"message":"Dataset with name \"new_dataset\" already exists"}]}})

To avoid error, use 'change_name_if_conflict' flag:

    new_dataset = api.dataset.create(53939, 'new_dataset', change_name_if_conflict=True)
    print(new_dataset)

    DatasetInfo(id=441022, name='new_dataset_001', description='', size='0', project_id=53939, images_count=0, items_count=0, created_at='2021-02-06T11:34:43.100Z', updated_at='2021-02-06T11:34:43.100Z', reference_image_url=None)

Get a dataset with a given name from a project with a given ID, if the dataset is not in the project, create it:

    dataset = api.dataset.get_or_create(53939, 'new_dataset_name')

Copy given list of datasets in destination project:

    new_datasets = api.dataset.copy_batch(53939, [365184], new_names=['new_ds'], change_name_if_conflict=True, with_annotations=True)
    print(new_datasets)

    [DatasetInfo(id=441031, name='new_ds', description='', size='0', project_id=53939, images_count=0, items_count=0, created_at='2021-02-06T11:54:48.919Z', updated_at='2021-02-06T11:54:48.919Z', reference_image_url=None)]

If lengh of datasets ids list != lengh of new_names list generate error RuntimeError

Copy given dataset in destination project:

    new_datasets = api.dataset.copy_batch(53939, 365184, new_names='new_ds', change_name_if_conflict=True)
    print(new_datasets)

    [DatasetInfo(id=441031, name='new_ds', description='', size='0', project_id=53939, images_count=0, items_count=0, created_at='2021-02-06T11:54:48.919Z', updated_at='2021-02-06T11:54:48.919Z', reference_image_url=None)]

Moves given list of datasets in destination project:

    new_datasets = api.dataset.move_batch(53939, [365184], new_names=['new_ds'], change_name_if_conflict=True, with_annotations=True)
    print(new_datasets)

    [DatasetInfo(id=441031, name='new_ds', description='', size='0', project_id=53939, images_count=0, items_count=0, created_at='2021-02-06T11:54:48.919Z', updated_at='2021-02-06T11:54:48.919Z', reference_image_url=None)]

Move given dataset in destination project:

    new_datasets = api.dataset.move(53939, 365184, new_names='new_ds', change_name_if_conflict=True)
    print(new_datasets)

    [DatasetInfo(id=441031, name='new_ds', description='', size='0', project_id=53939, images_count=0, items_count=0, created_at='2021-02-06T11:54:48.919Z', updated_at='2021-02-06T11:54:48.919Z', reference_image_url=None)]



## Projects and datasets on host

### Projects on host

Read project from host:

    project_path = os.path.join(os.getcwd(), 'lemons_test')
    project = sly.Project(project_path, sly.OpenMode.READ)
    print(project.name)

    lemons_test

Get project meta:
    
    print(project.meta)

    ProjectMeta:
    Object Classes
    +-------+--------+----------------+--------+
    |  Name | Shape  |     Color      | Hotkey |
    +-------+--------+----------------+--------+
    |  kiwi | Bitmap |  [255, 0, 0]   |        |
    | lemon | Bitmap | [81, 198, 170] |        |
    +-------+--------+----------------+--------+
    Tags
    +------+------------+-----------------+--------+---------------+--------------------+
    | Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +------+------------+-----------------+--------+---------------+--------------------+
    +------+------------+-----------------+--------+---------------+--------------------+

Set new project meta to project:

    new_project_meta = sly.ProjectMeta()
    project.set_meta(new_project_meta)
    print(project.meta)

    ProjectMeta:
    Object Classes
    +------+-------+-------+--------+
    | Name | Shape | Color | Hotkey |
    +------+-------+-------+--------+
    +------+-------+-------+--------+
    Tags
    +------+------------+-----------------+--------+---------------+--------------------+
    | Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +------+------------+-----------------+--------+---------------+--------------------+
    +------+------------+-----------------+--------+---------------+--------------------+

Check total number of items in project:

    print(project.total_items)

    6

Check parent directory of project:

    print(project.parent_dir)

    /home/andrew/alex_work/app_private/sdk_docs_examples

Make copy of project in given directory:

    new_project = project.copy_data(os.getcwd(), dst_name='new_project')
    print(new_project.name)

    new_project

Create new project, if project directory already exists and is not empty raise RuntimeError:

    new_project_path = os.path.join(os.getcwd(), 'new_project')
    try:
        new_project = sly.Project(new_project_path, sly.OpenMode.CREATE)
        print(new_project.name)
    except RuntimeError as error:
        print(error)

    new_project

### Datasets on host

Iteration between datasets in project:

    project_path = os.path.join(os.getcwd(), 'lemons_test')
    project = sly.Project(project_path, sly.OpenMode.READ)
    for dataset in project.datasets:
        print(dataset.name)

    ds1

Get path to images directory in dataset:
    
    for dataset in project.datasets:
        print(dataset.img_dir)

    /home/andrew/alex_work/app_private/sdk_docs_examples/lemons_test/ds1/img

Get path to annotations directory in dataset:
    
    for dataset in project.datasets:
        print(dataset.ann_dir)

    /home/andrew/alex_work/app_private/sdk_docs_examples/lemons_test/ds1/ann

Check that an image with a given name exists in the dataset:

    for dataset in project.datasets:
        print(dataset.item_exists('IMG_1836.jpeg'))
        print(dataset.item_exists('no_exist_image.jpeg'))

    True
    False

Get path to image with given name and it annotation, if image or annotation not exist RuntimeError is occured:

    for dataset in project.datasets:
        print(dataset.get_img_path('IMG_1836.jpeg'))
        print(dataset.get_ann_path('IMG_1836.jpeg'))

    /home/andrew/alex_work/app_private/sdk_docs_examples/lemons_test/ds1/img/IMG_1836.jpeg
    /home/andrew/alex_work/app_private/sdk_docs_examples/lemons_test/ds1/ann/IMG_1836.jpeg.json

Add image to dataset, empty annotation should be created automatically, if image not exist RuntimeError is occured:

    for dataset in project.datasets:
        dataset.add_item_file('IMG_0777.jpeg', os.path.join(os.getcwd(), 'new_image.jpeg'))
        print(dataset.item_exists('IMG_0777.jpeg'))
        print(dataset.get_ann_path('IMG_0777.jpeg'))

    True
    /home/andrew/alex_work/app_private/sdk_docs_examples/lemons_test/ds1/ann/IMG_0777.jpeg.json
    
Add image from numpy array to dataset, if image with given name already exists in dataset or new item name has unsupported extension RuntimeError is occured:

    image_np = sly.image.read(os.path.join(os.getcwd(), 'new_image.jpeg'))
    for dataset in project.datasets:
        dataset.add_item_np('IMG_0888.jpeg', image_np)
        print(dataset.item_exists('IMG_0888.jpeg'))

    True

Set annotation to given image in project, if image with given name not exist in dataset, raise RuntimeError:

    for dataset in project.datasets:
        dataset.set_ann_file('IMG_1836.jpeg', os.path.join(os.getcwd(), 'ann.json'))

Set annotation from sly.Annotation object, if image with given name not exist in dataset, raise RuntimeError:

    ann = sly.Annotation(img_size=(800, 1067))
    for dataset in project.datasets:
        dataset.set_ann('IMG_1836.jpeg', ann)

Set annotation from dict(json format), if image with given name not exist in dataset, raise RuntimeError:

    ann_json = {'size': {'height': 800, 'width': 1070}, 'tags': [], 'objects': []}
    for dataset in project.datasets:
        dataset.set_ann_dict('IMG_1836.jpeg', ann_json)

Create dataset in project:

    from supervisely_lib.project.project import Dataset
    dataset_path = os.path.join(project_path, 'ds_name')
    new_dataset = Dataset(dataset_path, sly.OpenMode.CREATE)
    print(new_dataset.name)

    ds_name

## ProjectMeta

Create empty ProjectMeta:

    project_meta = sly.ProjectMeta()
    print(project_meta)

    ProjectMeta:
    Object Classes
    +------+-------+-------+--------+
    | Name | Shape | Color | Hotkey |
    +------+-------+-------+--------+
    +------+-------+-------+--------+
    Tags
    +------+------------+-----------------+--------+---------------+--------------------+
    | Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +------+------------+-----------------+--------+---------------+--------------------+
    +------+------------+-----------------+--------+---------------+--------------------+

Add object class to ProjectMeta:

    obj_class_lemon = sly.ObjClass('lemon', sly.Rectangle)
    project_meta = project_meta.add_obj_class(obj_class_lemon)
    print(project_meta)

    ProjectMeta:
    Object Classes
    +-------+-----------+----------------+--------+
    |  Name |   Shape   |     Color      | Hotkey |
    +-------+-----------+----------------+--------+
    | lemon | Rectangle | [138, 109, 15] | lemon  |
    +-------+-----------+----------------+--------+
    Tags
    +------+------------+-----------------+--------+---------------+--------------------+
    | Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +------+------------+-----------------+--------+---------------+--------------------+
    +------+------------+-----------------+--------+---------------+--------------------+

Add object classes to ProjectMeta:

    obj_class_kiwi = sly.ObjClass('kiwi', sly.Bitmap)
    obj_class_cucumber = sly.ObjClass('cucumber', sly.Polygon)
    project_meta = project_meta.add_obj_classes([obj_class_kiwi, obj_class_cucumber])
    print(project_meta)

    ProjectMeta:
    Object Classes
    +----------+-----------+----------------+--------+
    |   Name   |   Shape   |     Color      | Hotkey |
    +----------+-----------+----------------+--------+
    |  lemon   | Rectangle | [63, 138, 15]  | lemon  |
    |   kiwi   |   Bitmap  | [138, 133, 15] |        |
    | cucumber |  Polygon  | [138, 87, 15]  |        |
    +----------+-----------+----------------+--------+
    Tags
    +------+------------+-----------------+--------+---------------+--------------------+
    | Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +------+------------+-----------------+--------+---------------+--------------------+
    +------+------------+-----------------+--------+---------------+--------------------+

Add tag meta to ProjectMeta:

    from supervisely_lib.annotation.tag_meta import TagApplicableTo
    tag_meta = sly.TagMeta('any_string_tag', sly.TagValueType.ANY_STRING, applicable_to=TagApplicableTo.IMAGES_ONLY)
    project_meta = project_meta.add_tag_meta(tag_meta)
    print(project_meta)

    ProjectMeta:
    Object Classes
    +----------+-----------+----------------+--------+
    |   Name   |   Shape   |     Color      | Hotkey |
    +----------+-----------+----------------+--------+
    |  lemon   | Rectangle | [15, 138, 135] | lemon  |
    |   kiwi   |   Bitmap  | [138, 97, 15]  |        |
    | cucumber |  Polygon  | [15, 138, 56]  |        |
    +----------+-----------+----------------+--------+
    Tags
    +----------------+------------+-----------------+--------+---------------+--------------------+
    |      Name      | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +----------------+------------+-----------------+--------+---------------+--------------------+
    | any_string_tag | any_string |       None      |        |   imagesOnly  |         []         |
    +----------------+------------+-----------------+--------+---------------+--------------------+

Add tag metas to ProjectMeta:

    tag_meta_number = sly.TagMeta('any_number_tag', sly.TagValueType.ANY_NUMBER)
    tag_meta_none = sly.TagMeta('None_tag', sly.TagValueType.NONE, applicable_to=TagApplicableTo.OBJECTS_ONLY)
    project_meta = project_meta.add_tag_metas([tag_meta_number, tag_meta_none])
    print(project_meta)

    ProjectMeta:
    Object Classes
    +----------+-----------+----------------+--------+
    |   Name   |   Shape   |     Color      | Hotkey |
    +----------+-----------+----------------+--------+
    |  lemon   | Rectangle | [138, 15, 52]  | lemon  |
    |   kiwi   |   Bitmap  | [15, 73, 138]  |        |
    | cucumber |  Polygon  | [138, 120, 15] |        |
    +----------+-----------+----------------+--------+
    Tags
    +----------------+------------+-----------------+--------+---------------+--------------------+
    |      Name      | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +----------------+------------+-----------------+--------+---------------+--------------------+
    | any_string_tag | any_string |       None      |        |   imagesOnly  |         []         |
    | any_number_tag | any_number |       None      |        |      all      |         []         |
    |    None_tag    |    none    |       None      |        |  objectsOnly  |         []         |
    +----------------+------------+-----------------+--------+---------------+--------------------+

Create ProjectMeta with classes and tags:

    obj_class_collection = sly.ObjClassCollection([obj_class_lemon, obj_class_kiwi, obj_class_cucumber])
    tag_metas_collection = sly.TagMetaCollection([tag_meta, tag_meta_number, tag_meta_none])
    project_meta_2 = sly.ProjectMeta(obj_classes=obj_class_collection, tag_metas=tag_metas_collection)
    print(project_meta_2)

    ProjectMeta:
    Object Classes
    +----------+-----------+----------------+--------+
    |   Name   |   Shape   |     Color      | Hotkey |
    +----------+-----------+----------------+--------+
    |  lemon   | Rectangle | [23, 15, 138]  | lemon  |
    |   kiwi   |   Bitmap  | [38, 15, 138]  |        |
    | cucumber |  Polygon  | [132, 15, 138] |        |
    +----------+-----------+----------------+--------+
    Tags
    +----------------+------------+-----------------+--------+---------------+--------------------+
    |      Name      | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +----------------+------------+-----------------+--------+---------------+--------------------+
    | any_string_tag | any_string |       None      |        |   imagesOnly  |         []         |
    | any_number_tag | any_number |       None      |        |      all      |         []         |
    |    None_tag    |    none    |       None      |        |  objectsOnly  |         []         |
    +----------------+------------+-----------------+--------+---------------+--------------------+

Merge ProjectMetas:

    project_meta_to_merge = sly.ProjectMeta()
    obj_class_apple = sly.ObjClass('apple', sly.Rectangle)
    project_meta_to_merge = project_meta_to_merge.add_obj_class(obj_class_apple)
    tag_meta = sly.TagMeta('merge_tag', sly.TagValueType.ANY_STRING)
    project_meta_to_merge = project_meta_to_merge.add_tag_meta(tag_meta)
    merge_meta = project_meta.merge(project_meta_to_merge)
    print(merge_meta)

    ProjectMeta:
    Object Classes
    +----------+-----------+----------------+--------+
    |   Name   |   Shape   |     Color      | Hotkey |
    +----------+-----------+----------------+--------+
    |  apple   | Rectangle | [113, 138, 15] |        |
    |  lemon   | Rectangle | [99, 138, 15]  | lemon  |
    |   kiwi   |   Bitmap  | [92, 15, 138]  |        |
    | cucumber |  Polygon  | [137, 15, 138] |        |
    +----------+-----------+----------------+--------+
    Tags
    +----------------+------------+-----------------+--------+---------------+--------------------+
    |      Name      | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +----------------+------------+-----------------+--------+---------------+--------------------+
    |   merge_tag    | any_string |       None      |        |      all      |         []         |
    | any_string_tag | any_string |       None      |        |   imagesOnly  |         []         |
    | any_number_tag | any_number |       None      |        |      all      |         []         |
    |    None_tag    |    none    |       None      |        |  objectsOnly  |         []         |
    +----------------+------------+-----------------+--------+---------------+--------------------+

Merge ProjectMetas with conflict class:

    conflict_meta = sly.ProjectMeta()
    obj_class_conflict = sly.ObjClass('lemon', sly.Polyline)
    conflict_meta = conflict_meta.add_obj_class(obj_class_conflict)
    try:
        conflict_meta = project_meta.merge(conflict_meta)
    except ValueError as error:
        print(error)

    Error during merge for key 'lemon': values are different

Merge ProjectMetas with conflict tag metas:

    tag_meta_conflict = sly.TagMeta('any_string_tag', sly.TagValueType.ANY_NUMBER)
    conflict_meta = conflict_meta.add_tag_meta(tag_meta_conflict)
    try:
        conflict_meta = project_meta.merge(conflict_meta)
    except ValueError as error:
        print(error)
    
    Error during merge for key 'any_string_tag': values are different

Get object class which is contained in ProjectMeta by it name:

    print(project_meta.get_obj_class('lemon'))

    Name:  lemon     Shape: Rectangle    Color: [15, 71, 138]  Geom. settings: {}              Hotkey ''

Get object class which is not contained in ProjectMeta by it name:

    print(project_meta.get_obj_class('pear'))

    None

Get tag meta which is contained in ProjectMeta:

    print(project_meta.get_tag_meta('any_number_tag'))

    Name:  any_number_tag           Value type:any_number    Possible values:None       Hotkey                  Applicable toall        Applicable classes[]

Get tag meta which is not contained in ProjectMeta:

    print(project_meta.get_tag_meta('some_tag_meta'))

    None

Add a object class to the ProjectMeta that is already contained there:

    from supervisely_lib.collection.key_indexed_collection import DuplicateKeyError
    try:
        project_meta = project_meta.add_obj_class(obj_class_conflict)
    except DuplicateKeyError as error:
        print(error)

    "Key 'lemon' already exists"

Add a tag meta to the ProjectMeta that is already contained there:

    try:
        project_meta = project_meta.add_tag_meta(tag_meta_conflict)
    except DuplicateKeyError as error:
        print(error)

    "Key 'any_string_tag' already exists"

Delete object class from ProjectMeta:

    project_meta = project_meta.delete_obj_class('lemon')
    print(project_meta)

    ProjectMeta:
    Object Classes
    +----------+---------+----------------+--------+
    |   Name   |  Shape  |     Color      | Hotkey |
    +----------+---------+----------------+--------+
    |   kiwi   |  Bitmap | [138, 60, 15]  |        |
    | cucumber | Polygon | [136, 138, 15] |        |
    +----------+---------+----------------+--------+
    Tags
    +----------------+------------+-----------------+--------+---------------+--------------------+
    |      Name      | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +----------------+------------+-----------------+--------+---------------+--------------------+
    | any_string_tag | any_string |       None      |        |   imagesOnly  |         []         |
    | any_number_tag | any_number |       None      |        |      all      |         []         |
    |    None_tag    |    none    |       None      |        |  objectsOnly  |         []         |
    +----------------+------------+-----------------+--------+---------------+--------------------+

Delete list of obj classes from ProjectMeta:

    project_meta = project_meta.delete_obj_classes(['kiwi', 'cucumber'])
    print(project_meta)

    ProjectMeta:
    Object Classes
    +------+-------+-------+--------+
    | Name | Shape | Color | Hotkey |
    +------+-------+-------+--------+
    +------+-------+-------+--------+
    Tags
    +----------------+------------+-----------------+--------+---------------+--------------------+
    |      Name      | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +----------------+------------+-----------------+--------+---------------+--------------------+
    | any_string_tag | any_string |       None      |        |   imagesOnly  |         []         |
    | any_number_tag | any_number |       None      |        |      all      |         []         |
    |    None_tag    |    none    |       None      |        |  objectsOnly  |         []         |
    +----------------+------------+-----------------+--------+---------------+--------------------+

Delete tag meta from ProjectMeta:

    project_meta = project_meta.delete_tag_meta('any_string_tag')
    print(project_meta)

    ProjectMeta:
    Object Classes
    +------+-------+-------+--------+
    | Name | Shape | Color | Hotkey |
    +------+-------+-------+--------+
    +------+-------+-------+--------+
    Tags
    +----------------+------------+-----------------+--------+---------------+--------------------+
    |      Name      | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +----------------+------------+-----------------+--------+---------------+--------------------+
    | any_number_tag | any_number |       None      |        |      all      |         []         |
    |    None_tag    |    none    |       None      |        |  objectsOnly  |         []         |
    +----------------+------------+-----------------+--------+---------------+--------------------+

Delete list of tag metas from ProjectMeta:

    project_meta = project_meta.delete_tag_metas(['any_number_tag', 'None_tag'])
    print(project_meta)

    ProjectMeta:
    Object Classes
    +------+-------+-------+--------+
    | Name | Shape | Color | Hotkey |
    +------+-------+-------+--------+
    +------+-------+-------+--------+
    Tags
    +------+------------+-----------------+--------+---------------+--------------------+
    | Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
    +------+------------+-----------------+--------+---------------+--------------------+
    +------+------------+-----------------+--------+---------------+--------------------+

