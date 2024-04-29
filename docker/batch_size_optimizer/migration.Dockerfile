FROM python:3.9

WORKDIR /workspace

ADD . /workspace

# For sqlite 
# RUN  pip install --no-cache-dir aiosqlite

# For mysql, we need asyncmy and cryptography (for sha256_password)
RUN pip install --no-cache-dir asyncmy cryptography 
RUN pip install --no-cache-dir '.[migration]'
