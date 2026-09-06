FROM supervisely/base-py-sdk:6.73.484

ARG tag_ref_name

RUN pip install albumentations==1.1.0

# Supervisely
RUN pip install --upgrade supervisely==$tag_ref_name
RUN pip install --upgrade supervisely[aug]==$tag_ref_name
