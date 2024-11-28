# Capriccio: A Drifting Sentiment Analysis Dataset

Capriccio is a drifting sentiment classification dataset on tweets.
It is created by slicing the Sentiment140 dataset ([homepage](http://help.sentiment140.com/home), [Huggingface datasets](https://huggingface.co/datasets/sentiment140)) with a sliding window of 500,000 tweets, resulting in 38 slices.
Thus, each slice can be used to represent the training/validation dataset of a sentiment classification model that is re-trained every day.
Each slice has 425,000 tweets for training (file named `%d_train.json`) and 75,000 tweets for validation (file named `%d_val.json`).

The name comes from the adjective *capricious*.

## Generating the dataset

```sh
pip install -r requirements.txt
python generate.py --output-dir data
```

Running `generate.py` will download the Sentiment140 dataset using Huggingface datasets.
Then, it will slice the dataset with a sliding window of size 500,000, ending up with 38 slices.
Finally, it will create train/validation splits for each slice and save them as JSON files in the directory pointed to by `--output-dir`.
For a slice, the train and validation sets are 48 MB and 8.4 MB respectively.

## Using the generated dataset

The generated slices (`%d_train.json` and `%d_val.json`) can be loaded into Huggingface with the following snippet:

```python
import datasets  # Huggingface datasets library

data_path = dict(train="9_train.json", validation="9_val.json")
raw_datasets = datasets.load_dataset("json", data_files=data_path)
```

For a full example, please refer to [`examples/batch_size_optimizer/capriccio`](/examples/batch_size_optimizer/capriccio).
