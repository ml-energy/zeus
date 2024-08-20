FROM mlenergy/zeus:v0.3.0

WORKDIR /workspace

ADD merak merak
RUN cd merak \
      && pip install pip==21.3 pybind11==2.10.3 \
      && pip install -e . \
      && cd ..

ADD lowtime lowtime
RUN cd lowtime \
      && pip install -e . \
      && cd ..

ADD perseus perseus
RUN cd perseus \
      && pip install -e . \
      && cd ..
