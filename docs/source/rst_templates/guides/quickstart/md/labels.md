
## Create and update labels

Create label:

    import supervisely_lib as sly
    label = sly.Label(sly.Rectangle(100, 100, 700, 900), sly.ObjClass('lemon', sly.Rectangle))
    print(label.to_json())

    {'classTitle': 'lemon', 'description': '', 'tags': [], 'points': {'exterior': [[100, 100], [900, 700]], 'interior': []}, 'geometryType': 'rectangle', 'shape': 'rectangle'}

Create label with tag:

    tag_meta = sly.TagMeta('any_string_tag', sly.TagValueType.ANY_STRING)
    tag = sly.Tag(tag_meta, 'Hello')
    label_with_tags = sly.Label(sly.Rectangle(100, 100, 700, 900), sly.ObjClass('lemon', sly.Rectangle), sly.TagCollection([tag]), 'label example')
    print(label_with_tags.to_json())

    {'classTitle': 'lemon', 'description': 'label example', 'tags': [{'name': 'any_string_tag', 'value': 'Hello'}], 'points': {'exterior': [[100, 100], [900, 700]], 'interior': []}, 'geometryType': 'rectangle', 'shape': 'rectangle'}

If try to create label with different types of object class geometry and type of figure in label raise RuntimeError:

    try:
        label = sly.Label(sly.Rectangle(100, 100, 700, 900), sly.ObjClass('lemon', sly.Bitmap))
    except RuntimeError as error:
        print(error)

    Input geometry type <class 'supervisely_lib.geometry.rectangle.Rectangle'> != geometry type of ObjClass <class 'supervisely_lib.geometry.bitmap.Bitmap'>

Create label from json format:

    project_meta = sly.ProjectMeta(sly.ObjClassCollection([sly.ObjClass('lemon', sly.Rectangle)]))
    label_json = {'classTitle': 'lemon', 'tags': [], 'points': {'exterior': [[100, 100], [900, 700]], 'interior': []}}
    label = sly.Label.from_json(label_json, project_meta)
    print(label.to_json())

    {'classTitle': 'lemon', 'description': '', 'tags': [], 'points': {'exterior': [[100, 100], [900, 700]], 'interior': []}, 'geometryType': 'rectangle', 'shape': 'rectangle'}

If given project meta not contain obj class with name, indicated in the json format, raise RuntimeError:

    try:
        project_meta = sly.ProjectMeta()
        label_json = {'classTitle': 'lemon', 'tags': [], 'points': {'exterior': [[100, 100], [900, 700]], 'interior': []}}
    except RuntimeError as error:
        print(error)

    Failed to deserialize a Label object from JSON: label class name 'lemon' was not found in the given project meta.

Get label area:

    print(label.area)

    481401.0

## Add tags to labels

Add tag to label:

    tag_meta = sly.TagMeta('any_number_tag', sly.TagValueType.ANY_NUMBER)
    tag = sly.Tag(tag_meta, 5)
    label_tags = label.add_tag(tag)
    print(label_tags.tags)

    Tags:
    +----------------+------------+-------+
    |      Name      | Value type | Value |
    +----------------+------------+-------+
    | any_number_tag | any_number |   5   |
    +----------------+------------+-------+

Add tags to label:

    tag_meta = sly.TagMeta('any_number_tag', sly.TagValueType.ANY_NUMBER)
    tag = sly.Tag(tag_meta, 5)
    tag_meta_none = sly.TagMeta('None_tag', sly.TagValueType.NONE)
    tag_none = sly.Tag(tag_meta_none)
    label_tags = label_tags.add_tags([tag, tag_none])
    print(label_tags.tags)

    Tags:
    +----------------+------------+-------+
    |      Name      | Value type | Value |
    +----------------+------------+-------+
    | any_number_tag | any_number |   5   |
    | any_number_tag | any_number |   5   |
    |    None_tag    |    none    |  None |
    +----------------+------------+-------+

NOTE! We can add same tag many times to label

## Labels transformations

Set image and annotation to make label transformations:

    meta_json = api.project.get_meta(116084)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(193605108)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    image_np = sly.image.read(os.path.join(os.getcwd(), 'lemons_test/example.jpeg'))
    print(ann.to_json())
    ann.draw(image_np)

    {'description': '', 'size': {'height': 800, 'width': 1067}, 'tags': [], 'objects': [{'classTitle': 'lemon', 'description': '', 'tags': [], 'bitmap': {'origin': [531, 120], 'data': 'eJwBZgOZ/IlQTkcNChoKAAAADUlIRFIAAAEZAAABBQEDAAAA7coBtAAAAAZQTFRFAAAA////pdmf3QAAAAF0Uk5TAEDm2GYAAAMOSURBVHiczdpLruMgEAVQWhlkyBLYSEssDS/NS/ESPMwgMt1J2qagPr4E66mZvME7iuFWmfgT5+Rxy/n5+hvyrIi3yX///fue80NF8YWeb5o1c89kTNYH7UOZ1I2avMooIKgyyvJuCPI1eoooAehXBtAdQQFBEUEZQDcEtVMSUUJQayTURimiNkqxwB5BEUB8SgLiUxLal09JQAFBiaPlKyQsjp/BrE8kJCTAkZAAR0ICHAmL40gwDEmLY0iaN0MRQdK8GZLMV0gqL0NiAm2rSOVlyCMoICgiKCFINA2SY2qQHFOD5Ji+QR5BQUYrkGWDEoJkU2+HSpY1UrKskRJT/Y3gERQQFBGkmAppCVRIS6BCWgIV8kNoA2KqUERQugxphiI1cHpdqwbejdSqUOQvQ+EyFBGUEKSbTmSUDkMTULpOZNS3E/lRhNS3E8VBVM679LPImPeBrE7pQ1an9CGrUyB07OP+MhRG0fFFHS9DyUBrFzJMuX5CkNUpx1X0OJqAdupDVjsdZ7kfRUf3BgP9jz23dqFkoOUytHeK2ZgTglwPGt4xn5ehx2VoRdByGZoRNHUh6yx3ANouQ08Elfs2Y+dZEbT0Ia+jGUFH4D9ymb0hiNy2JRU9EFQCN7aCBUElS6N7JwS5PqT33IYgkqXeTg8EkSy9ikiWQUUzgkqWRqcUM3gvTWLSq0Ji0gNfEURi0gMnMXkVkcUFzUAPL+iDpzSESAJ64CQBPXDSA3qWDkA0gbFHSnRxQUMLgkh59aqQBNTA6eLGHs/RxalZ0nmrWdJ5e2TeGoKeYlYPTaOCVgTRyqmBzwhyANoQ9EDQgqAJQM3LRuRoCnIAal9tIkcbeDfUGDGCDUHs9WcEpiSi1kiIv2wVUJuSeLYwIyD+QcLJyQ1H0m8O2IYhGLb1TCdodolVjSEZvEbZDuUfQHzGgVYELQZKZd4AmgwUzxdXkBLRZwQE7RW2YupD+k9zXOkVE+0VXi20V3hB0GyhvcKm+ZemebR9ebb59IFZlNfwZym9xu0spfc46bj9eGZ13R+H98AiRp9+EwAAAABJRU5ErkJgggn2cM0='}, 'shape': 'bitmap', 'geometryType': 'bitmap', 'labelerLogin': 'alexxx', 'updatedAt': '2021-02-10T08:39:24.828Z', 'createdAt': '2021-02-10T08:36:33.898Z', 'id': 619053179, 'classId': 2791451}, {'classTitle': 'kiwi', 'description': '', 'tags': [], 'points': {'exterior': [[764, 387], [967, 608]], 'interior': []}, 'labelerLogin': 'alexxx', 'updatedAt': '2021-02-10T08:39:24.828Z', 'createdAt': '2021-02-10T08:39:08.369Z', 'id': 619053347, 'classId': 2791453, 'geometryType': 'rectangle', 'shape': 'rectangle'}, {'classTitle': 'kiwi', 'description': '', 'tags': [], 'points': {'exterior': [[477, 543], [647, 713]], 'interior': []}, 'labelerLogin': 'alexxx', 'updatedAt': '2021-02-10T08:39:24.828Z', 'createdAt': '2021-02-10T08:39:16.938Z', 'id': 619053355, 'classId': 2791453, 'geometryType': 'rectangle', 'shape': 'rectangle'}, {'classTitle': 'cucumber', 'description': '', 'tags': [], 'points': {'exterior': [[140, 311], [185, 307], [226, 326], [292, 387], [349, 452], [397, 515], [396, 559], [356, 591], [322, 579], [213, 483], [136, 398], [121, 346]], 'interior': []}, 'shape': 'polygon', 'geometryType': 'polygon', 'labelerLogin': 'alexxx', 'updatedAt': '2021-02-10T08:39:39.577Z', 'createdAt': '2021-02-10T08:39:24.752Z', 'id': 619053360, 'classId': 2791454}], 'customBigData': {}}

<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/>


Crop label with given rectangle:

    for label in ann.labels:
    if label.obj_class.name == 'kiwi':
        ann = ann.delete_label(label)
        crop_labels = label.crop(sly.Rectangle(439, 550, 746, 850))
        for crop_label in crop_labels:
            ann = ann.add_label(crop_label)
    ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/gOo8W6J.jpg" alt="drawing" width="500"/>


Relative crop

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/0BXuXQm.jpg" alt="drawing" width="500"/>


Rotate label with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    for label in ann.labels:
    if label.obj_class.name == 'lemon':
        ann = ann.delete_label(label)
        rotate_label = label.rotate(ImageRotator(ann.img_size, 35))
        ann = ann.add_label(rotate_label)
    ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/frDgUp4.jpg" alt="drawing" width="500"/>


Resize label:

    for label in ann.labels:
    if label.obj_class.name == 'lemon':
        ann = ann.delete_label(label)
        resize_label = label.resize(ann.img_size, (600, 700))
        ann = ann.add_label(resize_label)
    ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/X79pFv3.jpg" alt="drawing" width="500"/>


Scale label:

    for label in ann.labels:
    if label.obj_class.name == 'lemon':
        ann = ann.delete_label(label)
        scale_label = label.scale(1.35)
        ann = ann.add_label(scale_label)
    ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/oAw1lWk.jpg" alt="drawing" width="500"/>


Translate label:

    for label in ann.labels:
        if label.obj_class.name == 'lemon':
            ann = ann.delete_label(label)
            translate_label = label.translate(250, -350)
            ann = ann.add_label(translate_label)
    ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/u0tyACF.jpg" alt="drawing" width="500"/>


Flip label from left to right:

    for label in ann.labels:
        if label.obj_class.name == 'lemon':
            ann = ann.delete_label(label)
            fliplr_label = label.fliplr(ann.img_size)
            ann = ann.add_label(fliplr_label)
    ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/K9y3omX.jpg" alt="drawing" width="500"/>


Flip label from up to down:

    for label in ann.labels:
        if label.obj_class.name == 'lemon':
            ann = ann.delete_label(label)
            flipud_label = label.flipud(ann.img_size)
            ann = ann.add_label(flipud_label)
    ann.draw(image_np)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/hF92laP.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/YwhPkB9.jpg" alt="drawing" width="500"/>


Convert label geometry to another(return list of geometries):

    for label in ann.labels:
    if label.obj_class.name == 'lemon':
        ann = ann.delete_label(label)
        label_to_rectangle = label.convert(sly.ObjClass('lemon', sly.Rectangle, color=label.obj_class.color))
        ann = ann.add_label(label_to_rectangle[0])
    ann.draw(image_np)

<img src="https://i.imgur.com/pasMWrq.jpg" alt="drawing" width="500"/> 


NOTE: possible to convert only bitmap, rectangle and polygon, in other way raise NotImplementedError

Draw label:

    mask = np.zeros((1000, 1200, 3), dtype=np.uint8)
    for label in ann.labels:
        if label.obj_class.name == 'lemon':
            label.draw(mask)

<img src="https://i.imgur.com/uc6otxK.jpg" alt="drawing" width="500"/> 


Draw label contour:

    mask = np.zeros((800, 1200, 3), dtype=np.uint8)
    for label in ann.labels:
        if label.obj_class.name == 'lemon':
            label.draw_contour(mask, [255, 255, 255], thickness=5)

<img src="https://i.imgur.com/k1YbA8j.jpg" alt="drawing" width="500"/> 

