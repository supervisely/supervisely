FROM supervisely/base-py-sdk:6.72.85

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install Cython==3.0.0 tensorboard==2.13.0
RUN pip install seaborn==0.12.2 ftfy==6.1.1 regex==2023.6.3
RUN pip install git+https://github.com/supervisely-ecosystem/depends-CLIP.git
RUN pip install imutils==0.5.4 loguru==0.7.0  
RUN pip install supervisely==6.72.85 supervisely[apps]==6.72.85

# No space left on device

# apply-nn-to-videos-project
# RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install Cython==3.0.0 tensorboard==2.13.0
# RUN pip install seaborn==0.12.2 ftfy==6.1.1 regex==2023.6.3
# RUN pip install git+https://github.com/supervisely-ecosystem/depends-CLIP.git
# RUN pip install supervisely==6.72.82

# # gl-metric-learning
# RUN pip install torch==1.11.0 pytorch_lightning==0.9.0
# RUN pip install scikit-learn==1.3.0 transformers==4.31.0 timm==0.9.2
# RUN pip install albumentations==1.1.0 apex==0.9.10dev
# RUN pip install supervisely==6.72.82 supervisely[apps]==6.72.82