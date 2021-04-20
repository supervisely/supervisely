# Pascal VOC Import

#### Usage steps:
1) Download `Pascal VOC` dataset from [official site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) ([direct dl Link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) or import your custom dataset.

2) Unpack archive

3) Directory structure have to be the following:

```text     
1) For Official Pascal VOC Dataset          2) For Custom Pascal VOC Dataset
.                                           .    
├── ImageSets                               ├── ImageSets
│   └── Segmentation                        │   └── Segmentation
│       ├── train.txt                       │       ├── train.txt
│       ├── trainval.txt                    │       ├── trainval.txt
│       └── val.txt                         │       └── val.txt
├── JPEGImages                              ├── JPEGImages
│   ├── 2007_000032.jpg                     │   ├── 2007_000032.jpg
│   ├── 2007_000033.jpg                     │   ├── 2007_000033.jpg
│   ├── ...                                 │   ├── ...
├── SegmentationClass                       ├── SegmentationClass
│   ├── 2007_000032.png                     │   ├── 2007_000032.png
│   ├── 2007_000033.png                     │   ├── 2007_000033.png
│   ├── ...                                 │   ├── ...
└── SegmentationObject                      |── SegmentationObject
    ├── 2007_000032.png                     |   ├── 2007_000032.png
    ├── 2007_000033.png                     |   ├── 2007_000033.png
    ├── ...                                 |   ├── ...
                                            |
                                            └── colors.txt
```         
**`colors.txt`** file is used to import custom Pascal VOC Datasets, this file is not provided in the original Pascal VOC Dataset. File contains information about instance mask colors associated with object classes. This file is required, if you are importing custom dataset in Pascal VOC format. If you are importing official Pascal VOC Dataset you don't need this file.

**`colors.txt`** example:
```text     
neutral 224 224 192
kiwi 255 0 0
lemon 81 198 170
```
        

4) Open [Supervisely import](supervise.ly/import) page. Choose `Pascal VOC` import plugin.

5) Select all subdirectories (`ImageSets`, `JPEGImages`, `SegmentationClass`, `SegmentationClass` and `colors.txt` if importing custom dataset) and drag and drop them to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `train`, `trainval`, `val`.

    ![](https://imgur.com/37jUZZ1.jpg)

8) Datasets samples contains images and `instance segmentation` annotations. See few examples:

    ![](https://i.imgur.com/hJ93iv3.jpg)
    
    ![](https://i.imgur.com/UVqlFlp.jpg)

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/BS79Qr2.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
