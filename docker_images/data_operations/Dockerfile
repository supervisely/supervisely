FROM supervisely/base-py-sdk:6.73.484

ARG tag_ref_name

RUN apt-get update -y
RUN apt-get install -y libzbar0

RUN pip install pyzbar==0.1.9
RUN pip install moviepy==1.0.3 imageio-ffmpeg==0.4.7
RUN pip install imutils==0.5.4 
RUN pip install decord==0.6.0

# Supervisely
RUN pip install --upgrade supervisely==$tag_ref_name
