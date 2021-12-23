
## Logging

Create logger:

    import supervisely_lib as sly
    logger = sly.logger
    print(logger)

    <Logger logger (INFO)>

Set log level and extra:

    logger.warn('text message', extra={"logger name": logger.name, 'event_type': sly.EventType.TASK_FINISHED})
    logger.info('text message')
    logger.error('text message', extra={"logger level": logger.level})
    logger.fatal('text message')

    {"message": "text message", "logger name": "logger", "event_type": "EventType.TASK_FINISHED", "timestamp": "2021-02-11T17:32:09.404Z", "level": "warn"}
    {"message": "text message", "timestamp": "2021-02-11T17:32:09.404Z", "level": "info"}
    {"message": "text message", "logger level": 20, "timestamp": "2021-02-11T17:32:09.404Z", "level": "error"}
    {"message": "text message", "timestamp": "2021-02-11T17:32:09.405Z", "level": "fatal"}

Types of log messages:

    for log_type, log_level in sly.sly_logger.LOGGING_LEVELS.items():
        print(log_type, log_level.descr, log_level.int)

    FATAL Critical error 50
    ERROR Error 40
    WARN Warning 30
    INFO Info 20
    DEBUG Debug 10
    TRACE Trace 5

Logging messages to file:

    from supervisely_lib.sly_logger import add_default_logging_into_file
    add_default_logging_into_file(logger, os.path.join(os.getcwd(), 'logging_dir'))

## Colors

Generate random RGB color:

    from supervisely_lib.imaging.color import random_rgb
    color = random_rgb()
    print(color)

    [138, 116, 15]

Generate new color which oppositely by exist colors:

    from supervisely_lib.imaging.color import generate_rgb
    color = generate_rgb([[255, 0, 255], [0, 0, 127]])
    print(color)

    [77, 138, 15]

Convert integer color format to HEX string:

    from supervisely_lib.imaging.color import rgb2hex
    color_hex = rgb2hex([255, 127, 64])
    print(color_hex)

    FF7F40

Convert HEX RGB string to integer RGB format:

    from supervisely_lib.imaging.color import hex2rgb
    color_rgb = hex2rgb('#FF7F40')
    print(color_rgb)

    [255, 127, 64]

## Files and folders

Extract file name from a given path:

    from supervisely_lib.io.fs import get_file_name
    print(get_file_name(os.path.join(os.getcwd(), 'lemons_test/example.json')))

    example

Extract file extension from a given path:

    from supervisely_lib.io.fs import get_file_ext    
    print(get_file_ext(os.path.join(os.getcwd(), 'lemons_test/example.json')))

    .json

Extract file name with ext from a given path:

    from supervisely_lib.io.fs import get_file_name_with_ext
    print(get_file_name_with_ext(os.path.join(os.getcwd(), 'lemons_test/example.json')))

    example.json

Recursively walk through folder and return list with all file paths:

    from supervisely_lib.io.fs import list_dir_recursively
    print(list_dir_recursively(os.path.join(os.getcwd(), 'lemons_test')))

    ['example.json', 'meta.json', 'example.jpeg', 'ds1/ann/IMG_4451.jpeg.json', 'ds1/ann/IMG_1836.jpeg.json', 'ds1/ann/IMG_2084.jpeg.json', 'ds1/ann/IMG_8144.jpeg.json', 'ds1/ann/IMG_0777.jpeg.json', 'ds1/ann/IMG_0888.jpeg.json', 'ds1/ann/IMG_0748.jpeg.json', 'ds1/ann/IMG_3861.jpeg.json', 'ds1/img/IMG_3861.jpeg', 'ds1/img/IMG_0748.jpeg', 'ds1/img/IMG_8144.jpeg', 'ds1/img/IMG_0777.jpeg', 'ds1/img/IMG_2084.jpeg', 'ds1/img/IMG_0888.jpeg', 'ds1/img/IMG_1836.jpeg', 'ds1/img/IMG_4451.jpeg']

Get list with file paths presented in given folder:

    from supervisely_lib.io.fs import list_files
    print(list_files(os.path.join(os.getcwd(), 'lemons_test')))

    ['/home/andrew/alex_work/app_private/sdk_docs_examples/lemons_test/example.json', '/home/andrew/alex_work/app_private/sdk_docs_examples/lemons_test/meta.json', '/home/andrew/alex_work/app_private/sdk_docs_examples/lemons_test/example.jpeg']

Check whether folder exists or not:

    from supervisely_lib.io.fs import dir_exists
    print(dir_exists(os.path.join(os.getcwd(), 'lemons_test')))

    True

Create a leaf folder:

    from supervisely_lib.io.fs import mkdir, dir_exists
    mkdir(os.path.join(os.getcwd(), 'new_dir'))
    print(dir_exists(os.path.join(os.getcwd(), 'new_dir')))

    True

Check whether folder is empty or not:

    from supervisely_lib.io.fs import dir_empty
    print(dir_empty(os.path.join(os.getcwd(), 'new_dir')))

    True

Check file exist:

    from supervisely_lib.io.fs import file_exists
    print(file_exists(os.path.join(os.getcwd(), 'lemons_test/example.json')))

    True

Copy file from one path to another:

    from supervisely_lib.io.fs import copy_file, file_exists
    copy_file(os.path.join(os.getcwd(), 'lemons_test/meta.json'), os.path.join(os.getcwd(), 'new_dir/copy_meta.json'))
    print(file_exists(os.path.join(os.getcwd(), 'new_dir/copy_meta.json')))

    True

Get list containing the names of the folders in the given folder:

    from supervisely_lib.io.fs import get_subdirs
    print(get_subdirs(os.path.join(os.getcwd())))

    ['new_dir', '.idea', '.git', 'venv', 'lemons_test']

Recursively delete a folder tree, but save root folder:

    from supervisely_lib.io.fs import clean_dir, list_dir_recursively
    print(list_dir_recursively(os.path.join(os.getcwd(), 'lemons_test')))
    clean_dir(os.path.join(os.path.join(os.getcwd(), 'lemons_test')))
    print(list_dir_recursively(os.path.join(os.getcwd(), 'lemons_test')))

    ['example.json', 'meta.json', 'example.jpeg', 'ds1/ann/IMG_4451.jpeg.json', 'ds1/ann/IMG_1836.jpeg.json', 'ds1/ann/IMG_2084.jpeg.json', 'ds1/ann/IMG_8144.jpeg.json', 'ds1/ann/IMG_0777.jpeg.json', 'ds1/ann/IMG_0888.jpeg.json', 'ds1/ann/IMG_0748.jpeg.json', 'ds1/ann/IMG_3861.jpeg.json', 'ds1/img/IMG_3861.jpeg', 'ds1/img/IMG_0748.jpeg', 'ds1/img/IMG_8144.jpeg', 'ds1/img/IMG_0777.jpeg', 'ds1/img/IMG_2084.jpeg', 'ds1/img/IMG_0888.jpeg', 'ds1/img/IMG_1836.jpeg', 'ds1/img/IMG_4451.jpeg']
    []

Get file size:

    from supervisely_lib.io.fs import get_file_size
    print(get_file_size(os.path.join(os.getcwd(), 'annotation.png')))

    16444

Get folder size:

    from supervisely_lib.io.fs import get_directory_size
    print(get_directory_size(os.path.join(os.getcwd(), 'lemons_test')))

    1500719

Get tar archive from folder:

    from supervisely_lib.io.fs import archive_directory, file_exists
    archive_directory(os.path.join(os.getcwd(), 'lemons_test'), os.path.join(os.getcwd(), 'new_archive.tar'))
    print(file_exists(os.path.join(os.getcwd(), 'new_archive.tar')))

## Json

Decode data from json file with given filename:

    from supervisely_lib.io.json import load_json_file
    data_json = load_json_file(os.path.join(os.getcwd(), 'ann.json'))
    print(data_json)


    {'description': '', 'tags': [], 'size': {'height': 800, 'width': 1067}, 'objects': [{'description': '', 'tags': [], 'classTitle': 'awesome-poly', 'points': {'exterior': [[569, 105], [568, 395], [777, 395], [777, 105]], 'interior': []}}]}

Write data in json format in file:

    from supervisely_lib.io.json import dump_json_file
    from supervisely_lib.io.fs import file_exists
    dump_json_file(data_json, os.path.join(os.getcwd(), 'new_ann.json'))
    print(file_exists(os.path.join(os.getcwd(), 'new_ann.json')))

    True
