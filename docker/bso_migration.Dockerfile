FROM python:3.9

WORKDIR /workspace

ADD . /workspace

# For sqlite 
# RUN  pip install --no-cache-dir aiosqlite

# For mysql 
RUN  pip install --no-cache-dir asyncmy
RUN  pip install --no-cache-dir '.[migration]'
