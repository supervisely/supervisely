FROM supervisely/base-py-sdk:6.72.85

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install Cython==3.0.0 tensorboard==2.13.0
RUN pip install seaborn==0.12.2 ftfy==6.1.1 regex==2023.6.3
RUN pip install git+https://github.com/supervisely-ecosystem/depends-CLIP.git
RUN pip install imutils==0.5.4 loguru==0.7.0

# Supervisely 
RUN pip install --upgrade supervisely==$tag_ref_name
RUN pip install --upgrade supervisely[apps]==$tag_ref_name
