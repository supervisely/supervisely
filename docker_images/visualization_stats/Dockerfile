FROM supervisely/base-py-sdk:6.72.85

ARG tag_ref_name

RUN pip install Cython==3.0.0 scikit-learn==1.3.0 
RUN pip install open3d==0.15.2 pycocotools==2.0.6 diskcache==5.2.1 seaborn==0.11.0

# Supervisely
RUN pip install --upgrade supervisely==$tag_ref_name
