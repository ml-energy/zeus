# Batch Size Optimizer in Zeus

## prerequisites

1. Install zeus batch size optimizer dependencies

    ```shell
    pip install '.[bso]'
    ```

2. Install additional dependencies for a chosen database
3. Install migration dependencies to use alembic for migration
4. Follow readme under migration to set up tables

5. Set up `.env` under `zeus/zeus/optimizer/batch_size/server`. Should specify, database_url, and database_password. For example,

    ```shell
    # Example .env file
    ZEUS_DATABASE_URL = "sqlite+aiosqlite:////home/Projects/zeus/test.db"
    ZEUS_DATABASE_PASSWORD = "SECRET"
    ZEUS_LOG_LEVEL = "INFO"
    ```

## Running

1. Run server

    ```shell
    # /zeus/zeus/optimizer/batch_size/server
    uvicorn router:app 
    ```

2. Run mnist example

    ```shell
    # /zeus/examples/bso_server
    python mnist.py
    ```
