FROM supervisely/base-py-sdk:6.72.85

ARG tag_ref_name

RUN pip install dropbox==11.36.2

# Supervisely
RUN pip install --upgrade supervisely==$tag_ref_name
