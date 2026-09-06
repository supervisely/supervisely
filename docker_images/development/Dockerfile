FROM supervisely/base-py-sdk:6.73.484

ARG tag_ref_name

RUN pip install art==5.7 htmllistparse==0.5.2

# Supervisely
RUN pip install --upgrade supervisely==$tag_ref_name
