FROM python:3.9

WORKDIR /workspace

# Copy necessary files 
# If I don't have this, pip install will complain ModuleNotFoundError: No module named zeus
ADD ./zeus/__init__.py /workspace/zeus/__init__.py
ADD ./zeus/exception.py /workspace/zeus/exception.py

ADD ./zeus/optimizer/batch_size /workspace/zeus/optimizer/batch_size

ADD ./zeus/util/logging.py /workspace/zeus/util/logging.py
ADD ./zeus/util/metric.py /workspace/zeus/util/metric.py
ADD ./zeus/util/pydantic_v1.py /workspace/zeus/util/pydantic_v1.py

ADD ./pyproject.toml /workspace

# For sqlite 
# RUN  pip install --no-cache-dir aiosqlite

# For mysql 
RUN  pip install --no-cache-dir asyncmy
RUN  pip install --no-cache-dir '.[bso-server]'

CMD ["uvicorn", "zeus.optimizer.batch_size.server.router:app", "--host", "0.0.0.0", "--port", "80"]