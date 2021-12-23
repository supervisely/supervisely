
## Create, download and upload annotations 

Create empty annotation:

    import supervisely_lib as sly
    annotation = sly.Annotation((300, 400))
    print(annotation.to_json())

    {'description': '', 'size': {'height': 300, 'width': 400}, 'tags': [], 'objects': [], 'customBigData': {}}

Input size of annotation should be tuple, in other way raise TypeError:

    try:
    annotation = sly.Annotation({300: 400})
    except TypeError as error:
        print(error)

    'img_size' has to be a tuple or a list. Given type "<class 'dict'>".

Create annotation with label:

    annotation = sly.Annotation((300, 400), [sly.Label(sly.Rectangle(100, 100, 700, 900), sly.ObjClass('lemon', sly.Rectangle))])
    print(annotation.to_json())

    {'description': '', 'size': {'height': 300, 'width': 400}, 'tags': [], 'objects': [{'classTitle': 'lemon', 'description': '', 'tags': [], 
    'points': {'exterior': [[100, 100], [399, 299]], 'interior': []}, 'geometryType': 'rectangle', 'shape': 'rectangle'}], 'customBigData': {}}

Create annotation with label and tag:

    tag_meta = sly.TagMeta('any_string_tag', sly.TagValueType.ANY_STRING)
    tag = sly.Tag(tag_meta, 'Hello')
    annotation_with_tag = sly.Annotation((300, 400),
                                [sly.Label(sly.Rectangle(100, 100, 700, 900), sly.ObjClass('lemon', sly.Rectangle))],
                                sly.TagCollection([tag]),
                                'example annotaion')
    print(annotation_with_tag.to_json())

    {'description': 'example annotaion', 'size': {'height': 300, 'width': 400}, 'tags': [{'name': 'any_string_tag', 'value': 'Hello'}], 
    'objects': [{'classTitle': 'lemon', 'description': '', 'tags': [], 'points': {'exterior': [[100, 100], [399, 299]], 'interior': []}, 
    'geometryType': 'rectangle', 'shape': 'rectangle'}], 'customBigData': {}}

Create annotation from json format:

    project_meta = sly.ProjectMeta(sly.ObjClassCollection([sly.ObjClass('lemon', sly.Rectangle)]))
    ann_json = {'description': '', 'size': {'height': 300, 'width': 400}, 'tags': [], 'objects': [{'classTitle': 'lemon', 'description': '', 'tags': [], 
    'points': {'exterior': [[100, 100], [399, 299]], 'interior': []}, 'geometryType': 'rectangle', 'shape': 'rectangle'}], 'customBigData': {}}
    annotation = sly.Annotation.from_json(ann_json, project_meta)
    print(annotation.to_json())

    {'description': '', 'size': {'height': 300, 'width': 400}, 'tags': [], 'objects': [{'classTitle': 'lemon', 'description': '', 'tags': [], 
    'points': {'exterior': [[100, 100], [399, 299]], 'interior': []}, 'geometryType': 'rectangle', 'shape': 'rectangle'}], 'customBigData': {}}

Create annotation from image on host:

    ann_from_image = sly.Annotation.from_img_path(os.path.join(os.getcwd(), 'new_image.jpeg'))
    print(ann_from_image.to_json())

    {'description': '', 'size': {'height': 800, 'width': 1067}, 'tags': [], 'objects': [], 'customBigData': {}}

Download annotaion from host:

    project_path = os.path.join(os.getcwd(), 'lemons_test')
    project = sly.Project(project_path, sly.OpenMode.READ)
    path_to_ann = os.path.join(os.getcwd(), 'lemons_test/ds1/ann/IMG_0777.jpeg.json')
    ann_from_host = sly.Annotation.load_json_file(path_to_ann, project.meta)
    print(ann_from_host.to_json())

    {'description': '', 'size': {'height': 800, 'width': 1067}, 'tags': [], 'objects': [], 'customBigData': {}}

Download annotaion from API:

    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)
    meta_json = api.project.get_meta(36135)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(93907659)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    print(ann.to_json())

    {'description': '', 'size': {'height': 800, 'width': 1067}, 'tags': [], 'objects': [{'classTitle': 'kiwi', 'description': '', 'tags': [], 
    'points': {'exterior': [[146, 428], [362, 609]], 'interior': []}, 'labelerLogin': 'alexxx', 'updatedAt': '2019-08-02T13:19:51.408Z', 
    'createdAt': '2019-08-02T13:19:47.376Z', 'id': 245673127, 'classId': 648330, 'geometryType': 'rectangle', 'shape': 'rectangle'}, 
    {'classTitle': 'kiwi', 'description': '', 'tags': [], 'points': {'exterior': [[724, 388], [952, 609]], 'interior': []}, 'labelerLogin': 'alexxx', 
    'updatedAt': '2019-08-02T13:19:51.260Z', 'createdAt': '2019-08-02T13:19:51.260Z', 'id': 245673133, 'classId': 648330, 'geometryType': 'rectangle', 
    'shape': 'rectangle'}], 'customBigData': {}}

Download batch of annotaions from API:

    dataset_id = 143202
    ann_infos = api.annotation.download_batch(dataset_id, [93907658, 93907659, 93907660])
    annotations = [sly.Annotation.from_json(ann_info.annotation, meta) for ann_info in ann_infos]

Get list of all the annotations in the selected dataset:

    anns_info = api.annotation.get_list(143202)
    for ann_info in anns_info:
        print(anns_info.annotation)

    {'description': '', 'tags': [], 'size': {'height': 800, 'width': 1067}, 'objects': [{'id': 245672751, 'classId': 648328, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:13:55.906Z', 'updatedAt': '2019-08-02T13:19:42.167Z', 'tags': [], 'classTitle': 'cucumber', 'points': {'exterior': [[136, 300], [494, 461]], 'interior': []}}, {'id': 245672761, 'classId': 648328, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:14:01.328Z', 'updatedAt': '2019-08-02T13:19:42.167Z', 'tags': [], 'classTitle': 'cucumber', 'points': {'exterior': [[583, 538], [879, 661]], 'interior': []}}, {'id': 245673110, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:19:32.502Z', 'updatedAt': '2019-08-02T13:19:42.167Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[640, 114], [815, 275]], 'interior': []}}, {'id': 245673114, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:19:36.922Z', 'updatedAt': '2019-08-02T13:19:42.167Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[617, 321], [804, 474]], 'interior': []}}, {'id': 245673119, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:19:41.910Z', 'updatedAt': '2019-08-02T13:19:41.910Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[352, 509], [500, 686]], 'interior': []}}]}
    {'description': '', 'tags': [], 'size': {'height': 800, 'width': 1067}, 'objects': [{'id': 245672710, 'classId': 648328, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:13:12.069Z', 'updatedAt': '2019-08-02T13:19:26.236Z', 'tags': [], 'classTitle': 'cucumber', 'points': {'exterior': [[57, 426], [205, 704]], 'interior': []}}, {'id': 245672716, 'classId': 648328, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:13:20.089Z', 'updatedAt': '2019-08-02T13:19:26.236Z', 'tags': [], 'classTitle': 'cucumber', 'points': {'exterior': [[745, 162], [980, 407]], 'interior': []}}, {'id': 245672724, 'classId': 648328, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:13:27.365Z', 'updatedAt': '2019-08-02T13:19:26.236Z', 'tags': [], 'classTitle': 'cucumber', 'points': {'exterior': [[535, 376], [762, 567]], 'interior': []}}, {'id': 245673061, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:18:39.142Z', 'updatedAt': '2019-08-02T13:19:26.236Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[248, 364], [380, 512]], 'interior': []}}, {'id': 245673070, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:18:49.949Z', 'updatedAt': '2019-08-02T13:19:26.236Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[728, 483], [893, 646]], 'interior': []}}, {'id': 245673082, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:19:02.578Z', 'updatedAt': '2019-08-02T13:19:26.236Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[407, 251], [604, 420]], 'interior': []}}, {'id': 245673096, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:19:15.071Z', 'updatedAt': '2019-08-02T13:19:26.236Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[323, 514], [495, 640]], 'interior': []}}, {'id': 245673099, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:19:20.391Z', 'updatedAt': '2019-08-02T13:19:26.236Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[433, 63], [598, 238]], 'interior': []}}, {'id': 245673105, 'classId': 648330, 'description': '', 'geometryType': 'rectangle', 'labelerLogin': 'alexxx', 'createdAt': '2019-08-02T13:19:25.908Z', 'updatedAt': '2019-08-02T13:19:25.908Z', 'tags': [], 'classTitle': 'kiwi', 'points': {'exterior': [[490, 633], [653, 760]], 'interior': []}}]}
    {'description': '', 'tags': [], 'size': {'height': 800, 'width': 1067}, 'objects': []}

Save annotation on host:

    from supervisely_lib.io.json import dump_json_file
    dump_json_file(ann.to_json(), 'lemons_test/ann/test.json')

Upload annotation on API from path:

    api.annotation.upload_path(93907658, 'lemons_test/ann/test.json')

If annotation file not exist raise FileNotFoundError.

Upload annotations on API from path:

    api.annotation.upload_paths([93907658, 93907659], ['lemons_test/ann/test.json', 'lemons_test/ann/ann_0777.json'])

If len(img_ids) != len(anns) raise RuntimeError, if annotation file not exist raise FileNotFoundError.

Upload annotation on API from json format:

    api.annotation.upload_json(93907658, ann.to_json())

Upload annotations on API from json format:

    api.annotation.upload_jsons([93907658, 93907659], [ann.to_json(), ann1.to_json()])

Upload annotation on API:

    api.annotation.upload_ann(93907658, ann)

Upload annotations on API:

    api.annotation.upload_anns([93907658, 93907659], [ann, ann1])

Copy annotation from one image to another on API, labels from src annotation must be in dst project meta, raise HTTPError in over way:

    api.annotation.copy(176024255, 193358498)

Copy annotations from one list of images to another on API, labels from src annotations must be in dst project meta, raise HTTPError in over way:

    api.annotation.copy_batch([176024256, 176024257], [193358499, 193358500])

NOTES:

1) If annotations dimensions' don't match images dimensions raise HTTPError!
   
2) If image ID does not exist raise Retry error!


## Add and delete labels and tags to annotation

Create empty annotation:

    annotation = sly.Annotation((1000, 1200))
    print(annotation.to_json())

    {'description': '', 'size': {'height': 300, 'width': 400}, 'tags': [], 'objects': [], 'customBigData': {}}

Add label to annotation:

    ann = annotation.add_label(sly.Label(sly.Point(100, 200), sly.ObjClass('new_label_name', sly.Point)))
    print(ann.to_json())

    {'description': '', 'size': {'height': 1000, 'width': 1200}, 'tags': [], 'objects': [{'classTitle': 'new_label_name', 'description': '', 'tags': [], 'points': {'exterior': [[200, 100]], 'interior': []}, 'geometryType': 'point', 'shape': 'point'}], 'customBigData': {}}

Add labels to annotation:

    new_ann = annotation.add_labels([sly.Label(sly.Point(100, 200), sly.ObjClass('marker', sly.Point)),
                                     sly.Label(sly.Point(300, 400), sly.ObjClass('second marker', sly.Point))])
    print(new_ann.to_json())

    {'description': '', 'size': {'height': 1000, 'width': 1200}, 'tags': [], 'objects': [{'classTitle': 'marker', 'description': '', 'tags': [], 'points': {'exterior': [[200, 100]], 'interior': []}, 'geometryType': 'point', 'shape': 'point'}, {'classTitle': 'second marker', 'description': '', 'tags': [], 'points': {'exterior': [[400, 300]], 'interior': []}, 'geometryType': 'point', 'shape': 'point'}], 'customBigData': {}}

Delete label from annotation by it name:

    for label in ann.labels:
    if label.obj_class.name == 'new_label_name':
        new_ann = annotation.delete_label(label)
    print(new_ann.to_json())

    {'description': '', 'size': {'height': 300, 'width': 400}, 'tags': [], 'objects': [], 'customBigData': {}}

If we try to remove a label from the annotation which is not there, raise KeyError:

    try:
        new_ann = annotation.delete_label(sly.Label(sly.Rectangle(100, 100, 700, 900), sly.ObjClass('kiwi', sly.Rectangle)))
    except KeyError as error:
        print(error)

    'Trying to delete a non-existing label of class: kiwi'

Add tag to annotation:

    tag_meta = sly.TagMeta('any_string_tag', sly.TagValueType.ANY_STRING)
    new_ann = annotation.add_tag(sly.Tag(tag_meta, 'Hello'))
    print(new_ann.to_json())

    {'description': '', 'size': {'height': 1000, 'width': 1200}, 'tags': [{'name': 'any_string_tag', 'value': 'Hello'}], 'objects': [], 'customBigData': {}}

Add tags to annotation:

    tag_meta = sly.TagMeta('any_string_tag', sly.TagValueType.ANY_STRING)
    tag_meta_none = sly.TagMeta('None_tag', sly.TagValueType.NONE)
    new_ann = annotation.add_tags([sly.Tag(tag_meta, 'String_tag'), sly.Tag(tag_meta_none)])
    print(new_ann.to_json())

    {'description': '', 'size': {'height': 1000, 'width': 1200}, 'tags': [{'name': 'any_string_tag', 'value': 'String_tag'}, {'name': 'None_tag'}], 'objects': [], 'customBigData': {}}

Delete tag by name from annotation:

    ann_del_tag = new_ann.delete_tag_by_name('any_string_tag')
    print(ann_del_tag.to_json())

    {'description': '', 'size': {'height': 1000, 'width': 1200}, 'tags': [{'name': 'None_tag'}], 'objects': [], 'customBigData': {}}

Delete tags by names from annotation:

    ann_del_tags = new_ann.delete_tags_by_name(['any_string_tag', 'None_tag'])
    print(ann_del_tags.to_json())

    {'description': '', 'size': {'height': 1000, 'width': 1200}, 'tags': [], 'objects': [], 'customBigData': {}}

Delete tag from annotation:

    for ann_tag in new_ann.img_tags:
    if ann_tag.name == 'any_string_tag':
        ann_del_tag = new_ann.delete_tag(ann_tag)
    print(ann_del_tag.to_json())

    {'description': '', 'size': {'height': 1000, 'width': 1200}, 'tags': [{'name': 'None_tag'}], 'objects': [], 'customBigData': {}}

Delete tags from annotation:

    tags_to_del = []
    for ann_tag in new_ann.img_tags:
        if ann_tag.name in ['any_string_tag', 'None_tag']:
            tags_to_del.append(ann_tag)
    ann_del_tags = new_ann.delete_tags(tags_to_del)
    print(ann_del_tags.to_json())

    {'description': '', 'size': {'height': 1000, 'width': 1200}, 'tags': [], 'objects': [], 'customBigData': {}}

Get statistics about classes in annotation:

    stat = annotation.stat_class_count(['lemon', 'kiwi', 'cucumber', 'some_class'])
    print(stat)

    {'lemon': 1, 'kiwi': 2, 'cucumber': 1, 'some_class': 0, 'total': 4}

NOTE: if the class that exists in annotation is not listed, raise KeyError

## Annotations transformations

Set image and annotation to make transformations:

    meta_json = api.project.get_meta(116084)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(193605108)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    image_np = sly.image.read(os.path.join(os.getcwd(), 'lemons_test/example.jpeg'))
    print(ann.to_json())
    ann.draw(image_np)

    {'description': '', 'size': {'height': 800, 'width': 1067}, 'tags': [], 'objects': [{'classTitle': 'lemon', 'description': '', 'tags': [], 'bitmap': {'origin': [531, 120], 'data': 'eJwBZgOZ/IlQTkcNChoKAAAADUlIRFIAAAEZAAABBQEDAAAA7coBtAAAAAZQTFRFAAAA////pdmf3QAAAAF0Uk5TAEDm2GYAAAMOSURBVHiczdpLruMgEAVQWhlkyBLYSEssDS/NS/ESPMwgMt1J2qagPr4E66mZvME7iuFWmfgT5+Rxy/n5+hvyrIi3yX///fue80NF8YWeb5o1c89kTNYH7UOZ1I2avMooIKgyyvJuCPI1eoooAehXBtAdQQFBEUEZQDcEtVMSUUJQayTURimiNkqxwB5BEUB8SgLiUxLal09JQAFBiaPlKyQsjp/BrE8kJCTAkZAAR0ICHAmL40gwDEmLY0iaN0MRQdK8GZLMV0gqL0NiAm2rSOVlyCMoICgiKCFINA2SY2qQHFOD5Ji+QR5BQUYrkGWDEoJkU2+HSpY1UrKskRJT/Y3gERQQFBGkmAppCVRIS6BCWgIV8kNoA2KqUERQugxphiI1cHpdqwbejdSqUOQvQ+EyFBGUEKSbTmSUDkMTULpOZNS3E/lRhNS3E8VBVM679LPImPeBrE7pQ1an9CGrUyB07OP+MhRG0fFFHS9DyUBrFzJMuX5CkNUpx1X0OJqAdupDVjsdZ7kfRUf3BgP9jz23dqFkoOUytHeK2ZgTglwPGt4xn5ehx2VoRdByGZoRNHUh6yx3ANouQ08Elfs2Y+dZEbT0Ia+jGUFH4D9ymb0hiNy2JRU9EFQCN7aCBUElS6N7JwS5PqT33IYgkqXeTg8EkSy9ikiWQUUzgkqWRqcUM3gvTWLSq0Ji0gNfEURi0gMnMXkVkcUFzUAPL+iDpzSESAJ64CQBPXDSA3qWDkA0gbFHSnRxQUMLgkh59aqQBNTA6eLGHs/RxalZ0nmrWdJ5e2TeGoKeYlYPTaOCVgTRyqmBzwhyANoQ9EDQgqAJQM3LRuRoCnIAal9tIkcbeDfUGDGCDUHs9WcEpiSi1kiIv2wVUJuSeLYwIyD+QcLJyQ1H0m8O2IYhGLb1TCdodolVjSEZvEbZDuUfQHzGgVYELQZKZd4AmgwUzxdXkBLRZwQE7RW2YupD+k9zXOkVE+0VXi20V3hB0GyhvcKm+ZemebR9ebb59IFZlNfwZym9xu0spfc46bj9eGZ13R+H98AiRp9+EwAAAABJRU5ErkJgggn2cM0='}, 'shape': 'bitmap', 'geometryType': 'bitmap', 'labelerLogin': 'alexxx', 'updatedAt': '2021-02-10T08:39:24.828Z', 'createdAt': '2021-02-10T08:36:33.898Z', 'id': 619053179, 'classId': 2791451}, {'classTitle': 'kiwi', 'description': '', 'tags': [], 'points': {'exterior': [[764, 387], [967, 608]], 'interior': []}, 'labelerLogin': 'alexxx', 'updatedAt': '2021-02-10T08:39:24.828Z', 'createdAt': '2021-02-10T08:39:08.369Z', 'id': 619053347, 'classId': 2791453, 'geometryType': 'rectangle', 'shape': 'rectangle'}, {'classTitle': 'kiwi', 'description': '', 'tags': [], 'points': {'exterior': [[477, 543], [647, 713]], 'interior': []}, 'labelerLogin': 'alexxx', 'updatedAt': '2021-02-10T08:39:24.828Z', 'createdAt': '2021-02-10T08:39:16.938Z', 'id': 619053355, 'classId': 2791453, 'geometryType': 'rectangle', 'shape': 'rectangle'}, {'classTitle': 'cucumber', 'description': '', 'tags': [], 'points': {'exterior': [[140, 311], [185, 307], [226, 326], [292, 387], [349, 452], [397, 515], [396, 559], [356, 591], [322, 579], [213, 483], [136, 398], [121, 346]], 'interior': []}, 'shape': 'polygon', 'geometryType': 'polygon', 'labelerLogin': 'alexxx', 'updatedAt': '2021-02-10T08:39:39.577Z', 'createdAt': '2021-02-10T08:39:24.752Z', 'id': 619053360, 'classId': 2791454}], 'customBigData': {}}

<img src="https://i.imgur.com/QgqWOdC.jpg" alt="drawing" width="500"/>

Crop annotation with given rectangle:

    crop_ann = annotation.crop_labels(sly.Rectangle(0, 0, 700, 700))
    crop_ann.draw(image_np)
    print(crop_ann.img_size)

    (800, 1067)

NOTE: image size in current Annotation object remained unchanged

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/QgqWOdC.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/3rwlmaZ.jpg" alt="drawing" width="500"/>


Relative_crop annotation with given rectangle:

    crop_ann = annotation.relative_crop(sly.Rectangle(0, 0, 700, 700))
    crop_ann.draw(image_np)
    print(crop_ann.img_size)

    (701, 701)

NOTE! image size in current Annotation object was changed

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/QgqWOdC.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/3rwlmaZ.jpg" alt="drawing" width="500"/>


Rotate with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    rotate_ann = annotation.rotate(ImageRotator(annotation.img_size, 25))
    rotate_ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/QgqWOdC.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/5fmF5go.jpg" alt="drawing" width="500"/>


Resize image with given annotation:

    resize_ann = annotation.resize((600, 700))
    resize_ann.draw(image_np)
    print(resize_ann.img_size)

    (600, 700)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/QgqWOdC.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/qwxu5IW.jpg" alt="drawing" width="375"/>


Scale annotation with given parameter(float):

    scale_ann = annotation.scale(0.55)
    scale_ann.draw(image_np)
    print(scale_ann.img_size)

    (440, 587)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/QgqWOdC.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/iUURKcU.jpg" alt="drawing" width="270"/>


Flip image with given annotation from left to right:

    fliplr_ann = annotation.fliplr()
    fliplr_ann.draw(image_np)


Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/QgqWOdC.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/9YCwH83.jpg" alt="drawing" width="500"/>


Flip image with given annotation from up to down:

    flipud_ann = annotation.flipud()
    flipud_ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/QgqWOdC.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/MiJZy5A.jpg" alt="drawing" width="500"/>


Draw annotation on mask:

    import cv2
    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    annotation.draw(mask)

<img src="https://i.imgur.com/MuezyX3.jpg" alt="drawing" width="500"/>

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    annotation.draw(mask, [128, 0 , 255])

<img src="https://i.imgur.com/OjyOL42.jpg" alt="drawing" width="500"/>


Draw annotation contour on mask:

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    annotation.draw_contour(mask, thickness=5)

<img src="https://i.imgur.com/MtXj3xB.jpg" alt="drawing" width="500"/>

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    annotation.draw_contour(mask, [255, 0, 0], thickness=5)

<img src="https://i.imgur.com/bpertWC.jpg" alt="drawing" width="500"/>


Draw annotation labels on mask with given indexes(colors):

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    name_to_index = {'lemon': 90, 'kiwi': 195, 'cucumber': 255}
    annotation.draw_class_idx_rgb(mask, name_to_index)

<img src="https://i.imgur.com/pr5Xle2.jpg" alt="drawing" width="500"/>


Filter annotation labels by min side:

    ann_filter = annotation.filter_labels_by_min_side(250)
    ann_filter.draw(image_np)

<img src="https://i.imgur.com/LYR4gz9.jpg" alt="drawing" width="500"/>

