FROM supervisely/base-py-sdk:6.73.137

ARG tag_ref_name

# Import
RUN pip install --no-build-isolation pycocotools==2.0.6
RUN pip install htmllistparse==0.6.0
RUN pip install google-cloud-storage==1.35.0 google-api-python-client==2.81.0
RUN pip install yt-dlp==2023.3.4 moviepy==1.0.3
RUN pip install open3d==0.15.2 scikit-learn==0.24.1
RUN pip install laspy==2.0.1 lazrs==0.3.0
RUN pip install git+https://github.com/autonomousvision/kitti360Scripts.git transforms3d==0.4.2
RUN pip install gdcm==1.1 pylibjpeg==1.4.0 pylibjpeg-libjpeg==1.3.1
RUN pip install pillow-avif-plugin==1.4.3 pillow-heif==0.18.0
RUN pip install tifffile==2023.7.10
RUN pip install pymupdf==1.22.5
RUN pip install nibabel==5.2.1
RUN pip install bagpy==0.5
RUN pip install imagecodecs==2023.3.16
RUN pip install lyft-dataset-sdk==0.0.8
RUN pip install nuscenes-devkit==1.1.11

# Export
RUN pip install lxml==4.6.3
RUN pip install scikit-image==0.18.3

# Supervisely
RUN pip install --upgrade supervisely==$tag_ref_name
