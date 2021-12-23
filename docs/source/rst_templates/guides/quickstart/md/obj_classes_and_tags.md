
## Object classes

Create object class:

    import supervisely_lib as sly
    obj_class = sly.ObjClass('lemon', sly.Rectangle)

    Name:  lemon     Shape: Rectangle    Color: [15, 20, 138]  Geom. settings: {}              Hotkey

Create object class with optional parameters:

    obj_class = sly.ObjClass('cucumber', sly.Bitmap, color=[128, 0, 255], geometry_config={'config':sly.Bitmap}, sly_id=123, hotkey='d')
    print(obj_class)
    print(obj_class.name)
    print(obj_class.geometry_type)
    print(obj_class.color)

    Name:  cucumber  Shape: Bitmap       Color: [128, 0, 255]  Geom. settings: {'config': <class 'supervisely_lib.geometry.bitmap.Bitmap'>}Hotkey d
    cucumber
    <class 'supervisely_lib.geometry.bitmap.Bitmap'>
    [128, 0, 255]

## Tags

Create any string tag:

    tag_meta = sly.TagMeta('any_string_tag', sly.TagValueType.ANY_STRING)
    tag = sly.Tag(tag_meta, 'Hello!')
    print(tag)
    print(tag.meta.color)
    print(tag.meta.applicable_to)

    Name:  any_string_tagValue type: any_string   Value:  Hello!  
    [15, 126, 138]
    all

NOTE: by default tags are applicable to both:labels and images

Create any number tag:

    tag_meta = sly.TagMeta('any_number_tag', sly.TagValueType.ANY_NUMBER, applicable_to=TagApplicableTo.ALL, color=[255, 0 ,128])
    tag = sly.Tag(tag_meta, 5)
    print(tag)
    print(tag.meta.color)
    print(tag.meta.applicable_to)

    Name:  any_number_tagValue type: any_number   Value:  5         
    [255, 0 ,128]
    all

Create None type tag:

    tag_meta = sly.TagMeta('None_tag', sly.TagValueType.NONE, color=[128, 0 ,255], applicable_to=TagApplicableTo.IMAGES_ONLY)
    tag = sly.Tag(tag_meta)
    print(tag)
    print(tag.meta.color)
    print(tag.meta.applicable_to)

    Name:  None_tag  Value type: none         Value:  None      
    [128, 0, 255]
    imagesOnly

Create one_of_string type tag:

    tag_meta = sly.TagMeta('oneof_string_tag', sly.TagValueType.ONEOF_STRING, possible_values=['lemon', 'kiwi', '123'], applicable_to=TagApplicableTo.OBJECTS_ONLY)
    tag = sly.Tag(tag_meta, 'lemon')
    print(tag)
    print(tag.meta.possible_values)
    print(tag.meta.applicable_to)

    Name:  oneof_string_tagValue type: oneof_string Value:  lemon     
    ['lemon', 'kiwi', '123']
    objectsOnly

##Possible errors when creating tags

Attempt to set integer value to any_string type tag

    tag_meta = sly.TagMeta('any_string_tag', sly.TagValueType.ANY_STRING)
    try:
        tag = sly.Tag(tag_meta, 777)
    except ValueError as error:
        print(error)

    Tag any_string_tag can not have value 777

Attempt to set string value to any_number type tag:

    tag_meta = sly.TagMeta('any_number_tag', sly.TagValueType.ANY_NUMBER, applicable_to=TagApplicableTo.ALL)
    try:
        tag = sly.Tag(tag_meta, 'cucumber')
    except ValueError as error:
        print(error)

    Tag any_number_tag can not have value cucumber

Attempt to set value to oneof_string type tag which is not in the list of valid values:

    tag_meta = sly.TagMeta('oneof_string_tag', sly.TagValueType.ONEOF_STRING, possible_values=['lemon', 'kiwi', '123'])
    try:
        tag = sly.Tag(tag_meta, 'cucumber')
    except ValueError as error:
        print(error)

    Tag oneof_string_tag can not have value cucumber

