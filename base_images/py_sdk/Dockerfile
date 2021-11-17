FROM supervisely/base-py:6.1.7

ARG tag_ref_name

RUN pip install attrs

#RUN pip install supervisely==6.2.2
RUN pip install supervisely==$(echo -n $tag_ref_name | cut -c2-)