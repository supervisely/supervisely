# Import Pointclouds

Just drag and drop the collection of ppintcloud files and folders with them. The new project will be created. 

Download [`pointcloud_without_ann_import_example.tar`](https://drive.google.com/file/d/1edi9hI6yG1MuIIPjw4aHrOIURX9WAbvj/view) example, unpack archive and drag-and-drop its content.  
   

## Datasets structure

Plugin creates datasets with names of top-most directories in a hierarchy. Files from root import directory will be placed to dataset with name "ds0".  

Let's consider several examples.
 
### Example 1. Import structure:

```
.
├── xxx.pcd
├── ds_ny
│   └── frame.pcd
└── ds_sf
    └── kitti_0000000001.pcd
```

In this case the following datasets will be created

- `ds_0` with a single file `xxx.pcd`
- `ds_ny` with a single file `frame.pcd`
- `ds_sf` with a single file `kitti_0000000001.pcd`


### Example 2. Import structure:

```
abcd_folder/
├── xxx.pcd
├── ds_ny
│   └── frame.pcd
└── ds_sf
    └── kitti_0000000001.pcd
```

In this case only the one dataset `abcd_folder` will be created with all pointcloud files.


### Example 3. PCD files with photo context:

Download [`pointcloud_with_photo_context.tar`](https://drive.google.com/file/d/1lZqzj6IE8rDYS2jnAAp6UaklpnSbNJWE/view?usp=sharing) example, unpack archive and drag-and-drop its content. 

```
abcd_folder/
└── dir_01
    └── dir_02
        ├── frame.pcd
        ├── kitti_0000000001.pcd
        └── related_images
            └── kitti_0000000001_pcd
                ├── 0000000000.png
                └── 0000000000.png.json
```

if you want to attach photo context to pcd file just create a directory `related_images` near the file. Then create directory `<filename_with_ext>` (in this example we name directory `kitti_0000000001_pcd` - it's a filename + extension + all symbols `.` are replaced to `_`) and put there images and corresponding `json` files with projection matrix. See example for more info.