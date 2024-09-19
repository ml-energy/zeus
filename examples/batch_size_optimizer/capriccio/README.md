# Capriccio + BSO

This example will demonstrate how to integrate Zeus with [Capriccio](../../research_reproducibility/zeus_nsdi23/capriccio), a drifting sentiment analysis dataset.

## Dependencies

1. Generate Capriccio, following the instructions in [Capriccio's README](../../research_reproducibility/zeus_nsdi23/capriccio).
1. Either use our Docker images or install `zeus` following our [documentation](https://ml.energy/zeus/getting_started/).
1. Install python dependencies for this example:
    ```sh
    pip install -r requirements.txt
    ```

## Running training

As described in the [MNIST example](../mnist/), set up the Zeus batch size optimizer server, and set the `ZEUS_SERVER_URL` environment variable.
On the first recurrence of the job, the batch size optimizer will register the job to the server and print out the job ID.
From the second recurrence, set the `ZEUS_JOB_ID` environment variable to allow the recurrence to be recognized as part of the recurring job.

```sh
python train.py \
    --data_dir data \
    --slice_number 9 \
    --model_name_or_path bert-base-uncased
```
