FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
RUN pip install tensorboardX

WORKDIR /workspace 

ADD . /workspace

RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -e '.[bso]'

RUN  chgrp -R 0 /workspace \
  && chmod -R g+rwX /workspace