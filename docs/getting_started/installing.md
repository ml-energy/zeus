# Installing Zeus

This document explains how to install the [`zeus`][zeus] Python package.

!!! Tip
    We encourage users to utilize our Docker image. Please refer to [Environment setup](./environment.md). Quick command:
    ```bash
    docker run -it --gpus all --cap-add SYS_ADMIN --ipc host mlenergy/zeus:latest bash
    ```


## `zeus` Python package

!!! Note
    This is already installed inside the container if you're using our Docker image.

### Install `zeus`

To install the latest stable release of `zeus`:

```bash
pip install zeus-ml
```

If you would like to follow `HEAD`:

```bash
git clone https://github.com/ml-energy/zeus.git zeus
cd zeus
pip install .
```

For those would like to make changes to the source code and run them, we suggest an editable installation:

```bash
git clone https://github.com/ml-energy/zeus.git zeus
cd zeus
pip install -e .
```
