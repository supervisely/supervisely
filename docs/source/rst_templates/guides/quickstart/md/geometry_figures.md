
## Point

### Create point

Create point:

    point = sly.Point(100, 200)
    print(point.to_json())
    print(point.col)
    print(point.row)
    print(point.area)

    {'points': {'exterior': [[200, 100]], 'interior': []}}
    200
    100
    0.0

Create pointlocation class object from point:

    location = point.point_location
    print(location.to_json())

    {'points': {'exterior': [[200, 100]], 'interior': []}}

Create point class object from pointlocation:

    point = sly.Point.from_point_location(location)
    print(point.to_json())

    {'points': {'exterior': [[200, 100]], 'interior': []}}

Create point from json format:

    point_json = {'points': {'exterior': [[10, 20]], 'interior': []}}
    point = sly.Point.from_json(point_json)
    print(point.to_json())

    {'points': {'exterior': [[10, 20]], 'interior': []}}

Create rectangle from point:

    rectangle = point.to_bbox()
    print(rectangle.to_json())

    {'points': {'exterior': [[200, 100], [200, 100]], 'interior': []}}

Draw point on mask:

    point = sly.Point(150, 150)
    mask = np.zeros((500, 500, 3), dtype=np.uint8)
    point.draw(mask, [255, 255, 255], thickness=7)    

<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/>

NOTE: draw_contour function will give the same result as draw.

### Point transformations

Crop with given rectangle:

    result = point.crop(sly.Rectangle(1, 1, 300, 350))
    for point in result:
        point.draw(mask, [255, 255, 255], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/> | <img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/>

    result = point.crop(sly.Rectangle(1, 1, 30, 35))
    for point in result:
        point.draw(mask, [255, 255, 255], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/> | <img src="https://i.imgur.com/yY7mpbB.jpg" alt="drawing" width="300"/>

Rotate with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    point = point.rotate(ImageRotator((300, 400), 45))
    point.draw(mask, [255, 255, 255], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/> | <img src="https://i.imgur.com/LLjDM7o.jpg" alt="drawing" width="300"/>

Resize image with given point:

    point = point.resize((300, 300), (150, 150))
    point.draw(mask, [255, 255, 255], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/> | <img src="https://i.imgur.com/w6e2xvM.jpg" alt="drawing" width="300"/>

Flip image with given point from left to right:

    point = point.fliplr((500, 500))
    point.draw(mask, [255, 255, 255], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/> | <img src="https://i.imgur.com/Og3eEB3.jpg" alt="drawing" width="300"/>

Flip image with given point from up to down:

    point = point.flipud((500, 500))
    point.draw(mask, [255, 255, 255], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/> | <img src="https://i.imgur.com/9SivhaV.jpg" alt="drawing" width="300"/>

Scale image with given point:

    point = point.scale(1.75)
    point.draw(mask, [255, 255, 255], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/> | <img src="https://i.imgur.com/vxh30Rc.jpg" alt="drawing" width="300"/>

Translate point:

    point = point.translate(320, 250)
    point.draw(mask, [255, 255, 255], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/UhTYYlB.jpg" alt="drawing" width="300"/> | <img src="https://i.imgur.com/UMmSnyC.jpg" alt="drawing" width="300"/>

## Bitmap

### Create bitmap

Create bitmap:

    mask = np.array([[0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 1, 0, 1, 0],
                     [0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0]], dtype=np.uint8)
    mask_bool = mask==1
    print(mask_bool)
    bitmap = sly.Bitmap(mask_bool)
    print(bitmap.to_json())
    print(bitmap.data)
    print(bitmap.area)
    print(bitmap.origin.to_json())

    [[False False False False False]
     [False  True  True  True False]
     [False  True False  True False]
     [False  True  True  True False]
     [False False False False False]]

    {'bitmap': {'origin': [1, 1], 'data': 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'}, 'shape': 'bitmap', 'geometryType': 'bitmap'}

     [[ True  True  True]
     [ True False  True]
     [ True  True  True]]

    8.0
    {'points': {'exterior': [[1, 1]], 'interior': []}}

If type of bitmap mask not bool raise ValueError:

    try:
        bitmap = sly.Bitmap(mask)
    except ValueError as error:
        print(error)

    Bitmap mask data must be a boolean numpy array. Instead got uint8.

If bitmap bool mask without True pixels raise ValueError:

    mask_full_false = mask==2
    print(mask_full_false)
    try:
        bitmap = sly.Bitmap(mask_full_false)
    except ValueError as error:
        print(error)

    [[False False False False False]
     [False False False False False]
     [False False False False False]
     [False False False False False]
     [False False False False False]]

    Creating a bitmap with an empty mask (no pixels set to True) is not supported.

Create bitmap from json format:

    bitmap_json = {'bitmap': {'origin': [1, 1], 'data': 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'}, 'shape': 'bitmap', 'geometryType': 'bitmap'}
    bitmap = sly.Bitmap.from_json(bitmap_json)
    print(bitmap.to_json())

    {'bitmap': {'origin': [1, 1], 'data': 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'}, 'shape': 'bitmap', 'geometryType': 'bitmap'}

### Bitmap transformations

Set bitmap from API to make transformations:

    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)
    meta_json = api.project.get_meta(116084)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(193605108)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    for label in ann.labels:
        if type(label.geometry) == sly.Bitmap:
            bitmap = label.geometry

Draw bitmap on mask:

    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    bitmap.draw(mask, color=[255, 0 , 128])

<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/>

Draw bitmap contour on mask:

    bitmap.draw_contour(mask, color=[255, 0 , 128], thickness=7)

<img src="https://i.imgur.com/Z5gP308.jpg" alt="drawing" width="500"/>

Crop with given rectangle:

    result = bitmap.crop(sly.Rectangle(1, 1, 450, 700))
    for bitmap in result:
        bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/UXxEQwh.jpg" alt="drawing" width="500"/>

Rotate with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    bitmap = bitmap.rotate(ImageRotator(ann.img_size, 45))
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/DvsrMnT.jpg" alt="drawing" width="500"/>

Resize image with given bitmap:

    bitmap = bitmap.resize(ann.img_size, (600, 800))
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/2A6NwgN.jpg" alt="drawing" width="500"/>

Translate bitmap:

    bitmap = bitmap.translate(350, 250)
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/ks2k2Pb.jpg" alt="drawing" width="500"/>

Flip image with given bitmap from left to right:

    bitmap = bitmap.fliplr((800, 1067))
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/EFj2f9u.jpg" alt="drawing" width="500"/>

Flip image with given bitmap from up to down:

    bitmap = bitmap.flipud((800, 1067))
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/BAGcL3v.jpg" alt="drawing" width="500"/>

Scale image with given bitmap:

    bitmap = bitmap.scale(0.65)
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/iZMykYt.jpg" alt="drawing" width="500"/>

Compute the skeleton of bitmap:

    from supervisely_lib.geometry.bitmap import SkeletonizeMethod
    bitmap = bitmap.skeletonize(SkeletonizeMethod.SKELETONIZE)
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/7jHHksC.jpg" alt="drawing" width="500"/>

Compute medial axis transform of bitmap:

    from supervisely_lib.geometry.bitmap import SkeletonizeMethod
    bitmap = bitmap.skeletonize(SkeletonizeMethod.MEDIAL_AXIS)
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/bo5vWx7.jpg" alt="drawing" width="500"/>

Compute morphological thinning of bitmap:

    from supervisely_lib.geometry.bitmap import SkeletonizeMethod
    bitmap = bitmap.skeletonize(SkeletonizeMethod.THINNING)
    bitmap.draw(mask, color=[255, 0 , 128])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/3ZqDGKK.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/7mA82uh.jpg" alt="drawing" width="500"/>

Get contours of bitmap(Polygon object):

    contours = bitmap.to_contours()
    for contour in contours:
        print(type(contour))

    <class 'supervisely_lib.geometry.polygon.Polygon'>

Bitwise operations with Bitmap:

Create second bitmap to make bitwise operations:

    second_mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    second_bitmap = bitmap.fliplr((800, 1067))
    second_bitmap = second_bitmap.translate(0, 200)
    second_bitmap.draw(second_mask, color=[255, 255 , 64])

<img src="https://i.imgur.com/P2lLg0a.jpg" alt="drawing" width="500"/>

Bitwise operation 'or' with Bitmap:

    bitwise_bitmap = bitmap.bitwise_mask(second_mask[:, :, 0], np.logical_or)
    bitwise_bitmap.draw(mask, color=[0, 255 , 64])

<img src="https://i.imgur.com/S3PokIy.jpg" alt="drawing" width="500"/>

Bitwise operation 'and' with Bitmap:

    bitwise_bitmap = bitmap.bitwise_mask(second_mask[:, :, 0], np.logical_and)
    bitwise_bitmap.draw(mask, color=[0, 255 , 64])

<img src="https://i.imgur.com/1z12gQ6.jpg" alt="drawing" width="500"/>

Bitwise operation 'xor' with Bitmap:

    bitwise_bitmap = bitmap.bitwise_mask(second_mask[:, :, 0], np.logical_xor)
    bitwise_bitmap.draw(mask, color=[0, 255 , 64])

<img src="https://i.imgur.com/88fhmdU.jpg" alt="drawing" width="500"/>

Create rectangle from bitmap:

    rectangle = bitmap.to_bbox()
    print(rectangle.to_json())

    {'points': {'exterior': [[531, 120], [811, 380]], 'interior': []}}

Convert bitmap to polygon:

    bitmap_to_polygon = bitmap.convert(sly.Polygon)
    for polygon in bitmap_to_polygon:
        print(type(polygon))

    <class 'supervisely_lib.geometry.polygon.Polygon'>

Convert bitmap data(bool numpy array) to base64 encoded string:

    encoded_string = sly.Bitmap.data_2_base64(bitmap.data)
    print(encoded_string)

    eJwBZgOZ/IlQTkcNChoKAAAADUlIRFIAAAEZAAABBQED...

Convert base64 encoded string to bitmap data(bool numpy array):

    bitmap_data = sly.Bitmap.base64_2_data(encoded_string)
    print(type(bitmap_data))

    <class 'numpy.ndarray'>

## Polygon

### Create polygon

Create polygon:

    import supervisely_lib as sly
    exterior = [sly.PointLocation(730, 2104), sly.PointLocation(2479, 402), sly.PointLocation(3746, 1646)]
    interior = [[sly.PointLocation(1907, 1255), sly.PointLocation(2468, 875), sly.PointLocation(2679, 1577)]]
    polygon = sly.Polygon(exterior, interior)
    print(polygon.to_json())
    print(polygon.exterior_np)

    {'points': {'exterior': [[2104, 730], [402, 2479], [1646, 3746]], 'interior': [[[1255, 1907], [875, 2468], [1577, 2679]]]}, 'shape': 'polygon', 'geometryType': 'polygon'}
     [[ 730 2104]
     [2479  402]
     [3746 1646]]

Create polygon from json format:

    polygon_json = {'points': {'exterior': [[2104, 730], [402, 2479], [1646, 3746]], 'interior': [[[1255, 1907], [875, 2468], [1577, 2679]]]}, 'shape': 'polygon', 'geometryType': 'polygon'}
    polygon = sly.Polygon.from_json(polygon_json)
    print(polygon.to_json())

    {'points': {'exterior': [[2104, 730], [402, 2479], [1646, 3746]], 'interior': [[[1255, 1907], [875, 2468], [1577, 2679]]]}, 'shape': 'polygon', 'geometryType': 'polygon'}

Get polygon area:

    print(polygon.area)

    2166095.0

Possible errors when creating polygon:

Polygon should contain at least 3 points:

    try:
        exterior = [sly.PointLocation(730, 2104), sly.PointLocation(2479, 402)]
        polygon = sly.Polygon(exterior, interior=[])
    except ValueError as error:
        print(error)

    "exterior" field must contain at least 3 points to create "Polygon" object.

Polygons interior should contain at least 3 points:

    try:
        exterior = [sly.PointLocation(730, 2104), sly.PointLocation(2479, 402), sly.PointLocation(3746, 1646)]
        interior = [[sly.PointLocation(1907, 1255), sly.PointLocation(2468, 875)]]
        polygon = sly.Polygon(exterior, interior)
    except ValueError as error:
        print(error)

    "interior" element must contain at least 3 points.

Polygon vertices must be PointLocation objects:

    try:
        exterior = [[730, 2104], [2479, 402], [3746, 1646]]
        polygon = sly.Polygon(exterior, interior=[])
    except TypeError as error:
        print(error)

    Argument "exterior" must be list of "PointLocation" objects!

### Polygon transformations

Set polygon from API to make transformations:

    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)
    meta_json = api.project.get_meta(116084)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(193605108)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    for label in ann.labels:
        if type(label.geometry) == sly.Polygon and len(label.geometry.interior) > 0:
            polygon = label.geometry

Draw polygon on mask:

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    polygon.draw(mask, [0, 215, 64])

<img src="https://i.imgur.com/RutRKjO.jpg" alt="drawing" width="500"/>

Draw polygon contour on mask:

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    polygon.draw_contour(mask, [0, 215, 64], thickness=5)

<img src="https://i.imgur.com/0NICxGF.jpg" alt="drawing" width="500"/>

Crop polygon with given rectangle:

    polygons = polygon.crop(sly.Rectangle(0, 0, 400, 470))
    for polygon in polygons:
        polygon.draw(mask, [0, 215, 64])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/RutRKjO.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/dIEDt4r.jpg" alt="drawing" width="500"/>

Resize image with given polygon:

    polygon = polygon.resize((800, 1067), (1500, 1800))
    polygon.draw(mask, [0, 215, 64])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/RutRKjO.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/cSRmw16.jpg" alt="drawing" width="500"/>

Scale image with given polygon:

    polygon = polygon.scale(1.5)
    polygon.draw(mask, [0, 215, 64])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/RutRKjO.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/npFNm70.jpg" alt="drawing" width="500"/>

Translate polygon:

    polygon = polygon.translate(350, -150)
    polygon.draw(mask, [0, 215, 64])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/RutRKjO.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/1aaeEvt.jpg" alt="drawing" width="500"/>

Rotate with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    polygon = polygon.rotate(ImageRotator((800, 1067), 45))
    polygon.draw(mask, [0, 215, 64])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/RutRKjO.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/wmRpH4b.jpg" alt="drawing" width="500"/>

Flip polygon from left to right:

    polygon = polygon.fliplr((800, 1067))
    polygon.draw(mask, [0, 215, 64])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/RutRKjO.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/OcupFtI.jpg" alt="drawing" width="500"/>

Flip polygon from up to down:

    polygon = polygon.flipud((800, 1067))
    polygon.draw(mask, [0, 215, 64])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/RutRKjO.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/q8Ys90A.jpg" alt="drawing" width="500"/>

Create rectangle from polygon:

    rectangle = polygon.to_bbox()
    rectangle.draw(mask, [0, 215, 64])

<img src="https://i.imgur.com/KBdBObQ.jpg" alt="drawing" width="500"/>

Convert polygon to bitmap:

    polygon_to_bitmap = polygon.convert(sly.Bitmap)
    print(type(polygon_to_bitmap[0]))

    <class 'supervisely_lib.geometry.bitmap.Bitmap'>

## Polyline

### Create polyline

Create polyline:

    exterior = [sly.PointLocation(730, 2104), sly.PointLocation(2479, 402), sly.PointLocation(1500, 780)]
    polyline = sly.Polyline(exterior)
    print('polyline in json format:',polyline.to_json())
    print(polyline.exterior_np)

    {'points': {'exterior': [[2104, 730], [402, 2479], [780, 1500]], 'interior': []}, 'shape': 'line', 'geometryType': 'line'}
     [[ 730 2104]
     [2479  402]
     [1500  780]]

Polyline should contain at least 2 points:

    try:
        polyline = sly.Polyline([sly.PointLocation(730, 2104)])
    except ValueError as error:
        print(error)

    "exterior" field must contain at least two points to create "Polyline" object.

Create polyline from json format:

    polyline_json = {'points': {'exterior': [[2104, 730], [402, 2479], [780, 1500]], 'interior': []}, 'shape': 'line', 'geometryType': 'line'}
    polyline = sly.Polyline.from_json(polyline_json)
    print(polyline.to_json())

    {'points': {'exterior': [[2104, 730], [402, 2479], [780, 1500]], 'interior': []}, 'shape': 'line', 'geometryType': 'line'}

Get polyline area:

    print(polyline.area)

    0.0

### Polyline transformations

Set polyline from API to make transformations:

    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)
    meta_json = api.project.get_meta(116084)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(193605108)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    for label in ann.labels:
        if type(label.geometry) == sly.Polyline:
            polyline = label.geometry

Draw polyline on mask:

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    polyline.draw(mask, [255, 64, 128], thickness=7)

<img src="https://i.imgur.com/NImmwcB.jpg" alt="drawing" width="500"/>

NOTE: draw_contour function will give the same result as draw.

Crop polyline with given rectangle:

    polylines = polyline.crop(sly.Rectangle(181, 215, 554, 806))
    for polyline in polylines:
        polyline.draw(mask, [255, 64, 128], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/NImmwcB.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/kG5K5Or.jpg" alt="drawing" width="500"/>

Resize polyline:

    polyline = polyline.resize((800, 1067), (400, 700))
    polyline.draw(mask, [255, 64, 128], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/NImmwcB.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/bjUpSva.jpg" alt="drawing" width="500"/>

Scale polyline:

    polyline = polyline.scale(1.75)
    polyline.draw(mask, [255, 64, 128], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/NImmwcB.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/5J7cawI.jpg" alt="drawing" width="500"/>

Translate polyline:

    polyline = polyline.translate(-250, 350)
    polyline.draw(mask, [255, 64, 128], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/NImmwcB.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/PYRAGo5.jpg" alt="drawing" width="500"/>

Rotate polyline with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    polyline = polyline.rotate(ImageRotator((800, 1067), 45))
    polyline.draw(mask, [255, 64, 128], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/NImmwcB.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/JjC57gt.jpg" alt="drawing" width="500"/>

Flip polyline from left to right:

    polyline = polyline.fliplr((800, 1067))
    polyline.draw(mask, [255, 64, 128], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/NImmwcB.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/fsBVRId.jpg" alt="drawing" width="500"/>

Flip polyline from up to down:

    polyline = polyline.flipud((800, 1067))
    polyline.draw(mask, [255, 64, 128], thickness=7)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/NImmwcB.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/UHp07gY.jpg" alt="drawing" width="500"/>

Create rectangle from polyline:

    rectangle = polyline.to_bbox()
    rectangle.draw(mask, [255, 64, 128], thickness=7)

<img src="https://i.imgur.com/4X3NDj7.jpg" alt="drawing" width="500"/>

Convert polyline to bitmap:

    polyline_to_bitmap = polyline.convert(sly.Bitmap)
    for bitmap in polyline_to_bitmap:
        print(type(bitmap))

    <class 'supervisely_lib.geometry.bitmap.Bitmap'>

Convert polyline to polygon:

    polyline_to_polygon = polyline.convert(sly.Polygon)
    for polygon in polyline_to_polygon:
        print(type(polygon))

    <class 'supervisely_lib.geometry.polygon.Polygon'>

If convert polyline from 2 points to polygon raise ValueError:

    try:
        polyline = sly.Polyline([sly.PointLocation(730, 2104), sly.PointLocation(2479, 402)])
        polyline_to_polygon = polyline.convert(sly.Polygon)
    except ValueError as error:
        print(error)

    "exterior" field must contain at least 3 points to create "Polygon" object.

## Rectangle

### Create rectangle, rectangle properties

Create rectangle:

    rectangle = sly.Rectangle(100, 100, 700, 900)
    print(rectangle.to_json())

    {'points': {'exterior': [[100, 100], [900, 700]], 'interior': []}}

Create rectangle from json format:

    rectangle_json = {'points': {'exterior': [[1, 1], [900, 700]], 'interior': []}}
    rectangle = sly.Rectangle.from_json(rectangle_json)
    print(rectangle.to_json())

    {'points': {'exterior': [[1, 1], [900, 700]], 'interior': []}}

Create rectangle from numpy array:

    rectangle_from_np = sly.Rectangle.from_array(np.zeros((300, 400)))
    print(rectangle_from_np.to_json())

    {'points': {'exterior': [[0, 0], [399, 299]], 'interior': []}}

Create rectangle with given size shape:

    rectangle_from_size = sly.Rectangle.from_size((300, 400))
    print(rectangle_from_size.to_json())

    {'points': {'exterior': [[0, 0], [399, 299]], 'interior': []}}

Create rectangle from given list of geometry type objects:

    geom_objs = [sly.Point(100, 200), sly.Polyline([sly.PointLocation(730, 2104), sly.PointLocation(2479, 402)])]
    rectangle_from_geom_objs = sly.Rectangle.from_geometries_list(geom_objs)
    print(rectangle_from_geom_objs.to_json())

    {'points': {'exterior': [[200, 100], [2104, 2479]], 'interior': []}}

Check rectangle contain given rectangle object:

    print(rectangle.contains(sly.Rectangle(200, 250, 400, 500)))
    print(rectangle.contains(sly.Rectangle(0, 0, 400, 500)))

    True
    False

Check rectangle contain given pointlocation object:

    print(rectangle.contains_point_location(sly.PointLocation(250, 300)))
    print(rectangle.contains_point_location(sly.PointLocation(1000, 3000)))

    True
    False

Convert rectangle to size:

    print(rectangle.to_size())

    (700, 900)

Checks slice of given numpy array with rectangle parameters:

    mask_slice = rectangle.get_cropped_numpy_slice(np.zeros((200, 500)))
    print(mask_slice.shape)

    (199, 499)

Checks intersects with another rectangle:

    print(rectangle.intersects_with(sly.Rectangle(90, 90, 400, 500)))
    print(rectangle.intersects_with(sly.Rectangle(1000, 1000, 4000, 5000)))

    True
    False

Get rectangle area:

    print(rectangle.area)

    481401.0

Get rectangle center:

    print(rectangle.center.to_json())

    {'points': {'exterior': [[500, 400]], 'interior': []}}

Get rectangle width and height:

    print(rectangle.width)
    print(rectangle.height)

    801
    601

Get rectangle corners:

    for corner in rectangle.corners:
        print(corner.to_json())

    {'points': {'exterior': [[1, 1]], 'interior': []}}
    {'points': {'exterior': [[900, 1]], 'interior': []}}
    {'points': {'exterior': [[900, 700]], 'interior': []}}
    {'points': {'exterior': [[1, 700]], 'interior': []}}

Possible error when creating rectangle:

    try:
        rectangle = sly.Rectangle(700, 1, 1, 900)
    except ValueError as error:
        print(error)

    Rectangle "top" argument must have less or equal value then "bottom"!

    try:
        rectangle = sly.Rectangle(1, 900, 700, 1)
    except ValueError as error:
        print(error)

    Rectangle "left" argument must have less or equal value then "right"!

### Rectangle transformations

Set rectangle from API to make transformations:

    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)
    meta_json = api.project.get_meta(116084)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(193605108)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    for label in ann.labels:
        if type(label.geometry) == sly.Rectangle:
            rectangle = label.geometry

Draw rectangle on mask:

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    rectangle.draw(mask, [255, 0, 0])

<img src="https://i.imgur.com/519C2Yp.jpg" alt="drawing" width="500"/>

Draw rectangle contour on mask:

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    rectangle.draw_contour(mask, [255, 0, 0], thickness=7)

<img src="https://i.imgur.com/5hjYy3o.jpg" alt="drawing" width="500"/>

Crop rectangle with given rectangle:

    result = rectangle.crop(sly.Rectangle(0, 0, 400, 450))
    for rectangle in result:
        rectangle.draw(mask, [255, 0, 0])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/519C2Yp.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/Fu8DFF0.jpg" alt="drawing" width="500"/>

Rotate with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    rectangle = rectangle.rotate(ImageRotator((800, 1067), 45))
    rectangle.draw(mask, [255, 0, 0])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/519C2Yp.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/7MZsMfY.jpg" alt="drawing" width="500"/>

Resize rectangle:

    rectangle = rectangle.resize((800, 1067), (500, 600))
    rectangle.draw(mask, [255, 0, 0])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/519C2Yp.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/Npy3SX9.jpg" alt="drawing" width="500"/>

Scale rectangle:

    rectangle = rectangle.scale(1.25)
    rectangle.draw(mask, [255, 0, 0])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/519C2Yp.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/TQPmUG6.jpg" alt="drawing" width="500"/>

Translate rectangle:

    rectangle = rectangle.translate(150, 250)
    rectangle.draw(mask, [255, 0, 0])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/519C2Yp.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/cPH3B66.jpg" alt="drawing" width="500"/>

Flip rectangle from left to right:

    rectangle = rectangle.fliplr((800, 1067))
    rectangle.draw(mask, [255, 0, 0])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/519C2Yp.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/2TRCQMP.jpg" alt="drawing" width="500"/>

Flip rectangle from up to down:

    rectangle = rectangle.flipud((800, 1067))
    rectangle.draw(mask, [255, 0, 0])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/519C2Yp.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/kQYG4HP.jpg" alt="drawing" width="500"/>

Convert rectangle to bitmap:

    rectangle_to_bitmap = rectangle.convert(sly.Bitmap)
    for bitmap in rectangle_to_bitmap:
        print(type(bitmap))

    <class 'supervisely_lib.geometry.bitmap.Bitmap'>

Convert rectangle to polygon:

    rectangle_to_polygon = rectangle.convert(sly.Polygon)
    for polygon in rectangle_to_polygon:
        print(type(polygon))    

    <class 'supervisely_lib.geometry.polygon.Polygon'>

## Cuboid

### Create cuboid, cuboid properties

Create cuboid:

    import supervisely_lib as sly
    from supervisely_lib.geometry.cuboid import CuboidFace
    faces = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 2)]
    points = [[277, 273], [840, 273], [840, 690], [277, 690], [688, 168], [1200, 168], [1200, 522]]
    pointlocation_points = [sly.PointLocation(point[0], point[1]) for point in points]
    cuboid = sly.Cuboid(pointlocation_points, faces)
    print('cuboid in json format', cuboid.to_json())

    {'points': [[273, 277], [273, 840], [690, 840], [690, 277], [168, 688], [168, 1200], [522, 1200]], 'faces': [[0, 1, 2, 3], [0, 4, 5, 1], [1, 5, 6, 2]]}

Create cuboid from json format:

    cuboid_json = {'points': [[273, 277], [273, 840], [690, 840], [690, 277], [168, 688], [168, 1200], [522, 1200]], 'faces': [[0, 1, 2, 3], [0, 4, 5, 1], [1, 5, 6, 2]]}
    cuboid = sly.Cuboid.from_json(cuboid_json)
    print(cuboid.to_json())

    {'points': [[273, 277], [273, 840], [690, 840], [690, 277], [168, 688], [168, 1200], [522, 1200]], 'faces': [[0, 1, 2, 3], [0, 4, 5, 1], [1, 5, 6, 2]]}

Possible error when creating cuboid:

    try:
        faces = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1)]
        cuboid = sly.Cuboid(pointlocation_points, faces)
    except ValueError as error:
        print(error)

    A cuboid must have exactly 3 faces. Instead got 2 faces.

    try:
        faces = [CuboidFace(0, 1, 2, 3), CuboidFace(0, 4, 5, 1), CuboidFace(1, 5, 6, 9)]
        cuboid = sly.Cuboid(pointlocation_points, faces)
    except ValueError as error:
        print(error)

    Point index is out of bounds for cuboid face. Got 7 points, but the index is 9.

Get cuboid area:

    print(cuboid.area)

    431298.0

### Cuboid transformations

Set cuboid from API to make transformations:

    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)
    meta_json = api.project.get_meta(116084)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(193605108)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    for label in ann.labels:
        if type(label.geometry) == sly.Cuboid:
            cuboid = label.geometry

Draw cuboid on mask:

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    cuboid.draw(mask, [128, 64, 255])

<img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/>

Draw cuboid contour on mask:

    mask = np.zeros((800, 1067, 3), dtype=np.uint8)
    cuboid.draw_contour(mask, [128, 64, 255], thickness=7)

<img src="https://i.imgur.com/N66rSFp.jpg" alt="drawing" width="500"/>

Crop cuboid with given rectangle:

    result = cuboid.crop(sly.Rectangle(1, 1, 700, 1000))
    for cuboid in result:
        cuboid.draw(mask, [128, 64, 255])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/>

NOTE: crop return list with Cuboid class object if rectangle contain all points of Cuboid and empty list otherwise

Rotate with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    cuboid = cuboid.rotate(ImageRotator((300, 400), 45))
    cuboid.draw(mask, [128, 64, 255])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/Z2VorTl.jpg" alt="drawing" width="500"/>

Resize cuboid:

    cuboid = cuboid.resize((800, 1067), (1000, 1300))
    cuboid.draw(mask, [128, 64, 255])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/7vbgsmf.jpg" alt="drawing" width="500"/>

Scale cuboid:

    cuboid = cuboid.scale(0.75)
    cuboid.draw(mask, [128, 64, 255])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/cPAeDds.jpg" alt="drawing" width="500"/>

Translate cuboid:

    cuboid = cuboid.translate(250, -250)
    cuboid.draw(mask, [128, 64, 255])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/qmq0kwg.jpg" alt="drawing" width="500"/>

Flip cuboid from left to right:

    cuboid = cuboid.fliplr((800, 1067))
    cuboid.draw(mask, [128, 64, 255])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/buojmE7.jpg" alt="drawing" width="500"/>

Flip cuboid from up to down:

    cuboid = cuboid.flipud((800, 1067))
    cuboid.draw(mask, [128, 64, 255])

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/sb4s2yJ.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/67Fgkn4.jpg" alt="drawing" width="500"/>

Create rectangle from cuboid:

    rectangle = cuboid.to_bbox()
    rectangle.draw(mask, [128, 64, 255])

<img src="https://i.imgur.com/FSyVRxz.jpg" alt="drawing" width="500"/>


## Point 3D

Create point 3D:

    from supervisely_lib.geometry.cuboid_3d import Vector3d
    from supervisely_lib.geometry.point_3d import Point3d
    vector_3d = Vector3d(5, 10, 15)
    point_3d = Point3d(vector_3d)
    print(point_3d.to_json())

    {'x': 5, 'y': 10, 'z': 15}

Create point_3d from json format:

    point_3d_json = {'x': 5, 'y': 10, 'z': 15}
    point_3d = Point3d.from_json(point_3d_json)
    print(point_3d.to_json())

    {'x': 5, 'y': 10, 'z': 15}

## Cuboid 3D

Create cuboid 3D:

    from supervisely_lib.geometry.cuboid_3d import Cuboid3d
    cuboid_3d = Cuboid3d(Vector3d(5, 10, 15), Vector3d(1, 2, 1), Vector3d(1, 2, 3))
    print(cuboid_3d.to_json())

    {'position': {'x': 5, 'y': 10, 'z': 15}, 'rotation': {'x': 1, 'y': 2, 'z': 1}, 'dimensions': {'x': 1, 'y': 2, 'z': 3}}

Create cuboid_3d from json format:

    cuboid_3d_json = {'position': {'x': 5, 'y': 10, 'z': 15}, 'rotation': {'x': 1, 'y': 2, 'z': 1}, 'dimensions': {'x': 1, 'y': 2, 'z': 3}}
    cuboid_3d = Cuboid3d.from_json(cuboid_3d_json)
    print(cuboid_3d.to_json())

    {'position': {'x': 5, 'y': 10, 'z': 15}, 'rotation': {'x': 1, 'y': 2, 'z': 1}, 'dimensions': {'x': 1, 'y': 2, 'z': 3}}

## Graph

### Create graph

    import supervisely_lib as sly
    from supervisely_lib.geometry.graph import Node, GraphNodes
    nodes = {0: Node(sly.PointLocation(5, 5), False), 1: Node(sly.PointLocation(100, 100), False), 2:Node(sly.PointLocation(200, 250), False)}
    graph = GraphNodes(nodes)
    print(graph.to_json())

    {'nodes': {0: {'loc': [5, 5]}, 1: {'loc': [100, 100]}, 2: {'loc': [250, 200]}}}

Create graph from json format:

    graph_json = {'nodes': {0: {'loc': [0, 0], 'disabled': True}, 1: {'loc': [1, 1], 'disabled': True}, 2: {'loc': [2, 2], 'disabled': True}}}
    graph = GraphNodes.from_json(graph_json)
    print(graph)

    {'nodes': {0: {'loc': [0, 0], 'disabled': True}, 1: {'loc': [1, 1], 'disabled': True}, 2: {'loc': [2, 2], 'disabled': True}}, 'classId': <class 'supervisely_lib.geometry.graph.Node'>}

Get graph area:

    print(graph.area)

    0.0

### Graph transformations

Set graph from API to make transformations:

    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)
    meta_json = api.project.get_meta(116084)
    meta = sly.ProjectMeta.from_json(meta_json)
    ann_info = api.annotation.download(193605108)
    ann = sly.Annotation.from_json(ann_info.annotation, meta)
    for label in ann.labels:
        if type(label.geometry) == GraphNodes:
            graph = label.geometry

Draw graph on mask:

    mask = np.zeros((700, 800, 3), dtype=np.uint8)
    graph.draw(mask, [255, 255, 0], thickness=10)

<img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/>

NOTE: draw_contour function will give the same result as draw.

Crop graph with given rectangle:

    crop_graph = graph.crop(sly.Rectangle(0, 0, 650, 750))
    for graph in crop_graph:
        graph.draw(mask, [255, 255, 0], thickness=10)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/>

NOTE: crop return list with Graph class object if rectangle contain all Graph object points and empty list otherwise

Translate graph:

    graph = graph.translate(150, 250)
    graph.draw(mask, [255, 255, 0], thickness=10)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/zq6XPvk.jpg" alt="drawing" width="500"/>

Resize graph:

    graph = graph.resize((700, 800), (350, 400))
    graph.draw(mask, [255, 255, 0], thickness=10)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/vNNxGth.jpg" alt="drawing" width="500"/>

Scale graph:

    graph = graph.scale(1.12)
    graph.draw(mask, [255, 255, 0], thickness=10)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/A0YXZsV.jpg" alt="drawing" width="500"/>

Rotate graph with given rotator:

    from supervisely_lib.geometry.image_rotator import ImageRotator
    graph = graph.rotate(ImageRotator((700, 800), 25))
    graph.draw(mask, [255, 255, 0], thickness=10)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/n60Rggp.jpg" alt="drawing" width="500"/>

Flip graph from left to right:

    graph = graph.fliplr((700, 800))
    graph.draw(mask, [255, 255, 0], thickness=10)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/BlukfId.jpg" alt="drawing" width="500"/>

Flip graph from up to down:

    graph = graph.flipud((700, 800))
    graph.draw(mask, [255, 255, 0], thickness=10)

Before  |  After
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/IGHF90T.jpg" alt="drawing" width="500"/> | <img src="https://i.imgur.com/pChyjw1.jpg" alt="drawing" width="500"/>

Create rectangle from graph:

    rectangle = graph.to_bbox()
    rectangle.draw(mask, [255, 255, 0])

![](https://i.imgur.com/nshEx5v.jpg)
