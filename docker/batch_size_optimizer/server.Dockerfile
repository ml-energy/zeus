FROM python:3.9

WORKDIR /workspace

ADD . /workspace

# For sqlite 
# RUN  pip install --no-cache-dir aiosqlite

# For mysql 
RUN pip install --no-cache-dir asyncmy cryptography 
RUN pip install --no-cache-dir '.[bso-server]'

CMD ["uvicorn", "zeus.optimizer.batch_size.server.router:app", "--host", "0.0.0.0", "--port", "80"]
