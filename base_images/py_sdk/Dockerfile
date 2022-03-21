FROM supervisely/base-py:6.1.7

ARG tag_ref_name

RUN pip install attrs

RUN pip install supervisely==$tag_ref_name
RUN pip install supervisely[apps]==$tag_ref_name