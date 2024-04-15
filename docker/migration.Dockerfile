FROM python:3.9

WORKDIR /workspace

# Add necessary files
ADD ./zeus/__init__.py /workspace/zeus/__init__.py

# Alembic files 
ADD ./zeus/optimizer/batch_size/migrations /workspace/zeus/optimizer/batch_size/migrations
ADD ./zeus/optimizer/batch_size/alembic.ini /workspace/zeus/optimizer/batch_size/alembic.ini

# copy files that is used by import of alembic files
ADD ./zeus/optimizer/batch_size/server/config.py /workspace/zeus/optimizer/batch_size/server/config.py

ADD ./zeus/optimizer/batch_size/server/database/schema.py /workspace/zeus/optimizer/batch_size/server/database/schema.py
ADD ./zeus/optimizer/batch_size/server/job/models.py /workspace/zeus/optimizer/batch_size/server/job/models.py
ADD ./zeus/optimizer/batch_size/common.py /workspace/zeus/optimizer/batch_size/common.py
ADD ./zeus/util/pydantic_v1.py /workspace/zeus/util/pydantic_v1.py

ADD ./pyproject.toml /workspace

# For sqlite 
# RUN  pip install --no-cache-dir aiosqlite

# For mysql 
RUN  pip install --no-cache-dir asyncmy
RUN  pip install --no-cache-dir '.[migration]'
