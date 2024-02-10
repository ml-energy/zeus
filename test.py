from zeus.monitor import ZeusMonitor
from zeus.optimizer import GlobalPowerLimitOptimizer
from zeus.optimizer.power_limit import HFGlobalPowerLimitOptimizer
import inspect
import unittest
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import torch
from datasets import load_dataset, load_metric

# TESTCASES: HFGlobalPowerLimitOptimizer
class TestOptimizerSignatures(unittest.TestCase):
    def test_constructor_signature_equality(self):
        """Ensure that the constructor signatures of GPLO and HFGPLO are exactly the same."""
        gplo_signature = inspect.signature(GlobalPowerLimitOptimizer.__init__)
        hfgplo_signature = inspect.signature(HFGlobalPowerLimitOptimizer.__init__)

        self.assertEqual(gplo_signature, hfgplo_signature, "Constructor signatures do not match.")

    def test_HFGPLO_inherits_TrainerCallback(self):
        """Ensure that HFGPLO inherits from TrainerCallback."""
        self.assertTrue(issubclass(HFGlobalPowerLimitOptimizer, TrainerCallback), "HFGPLO does not inherit from TrainerCallback.")
    
    def return_train_dataset(self, tokenizer):
        dataset = load_dataset("glue", "sst2")
        def tokenize_function(example):
            return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        train_dataset = tokenized_datasets["train"]

        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return train_dataset


    def test_HFGPLO_usage_single_GPU(self):
        """Ensure that HFGPLO can be used as a HuggingFace TrainerCallback. Single GPU Test."""
        monitor = ZeusMonitor(gpu_indices=[0])
        hfgplo = HFGlobalPowerLimitOptimizer(monitor)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        # Prepare dataset
        dataset = self.return_train_dataset(tokenizer)

        training_args = TrainingArguments(
            output_dir="./results/singlegpu",          # Output directory
            num_train_epochs=1,              # Total number of training epochs
            per_device_train_batch_size=8,   # Batch size per device during training
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[hfgplo]
        )

        trainer.train()

    def test_HFGPLO_usage_multi_GPU(self):
        """Ensure that HFGPLO can be used as a HuggingFace TrainerCallback. Multi GPU Test."""
        monitor = ZeusMonitor(gpu_indices=[0, 1, 2, 3])
        hfgplo = HFGlobalPowerLimitOptimizer(monitor)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        # Prepare dataset
        dataset = self.return_train_dataset(tokenizer)

        training_args = TrainingArguments(
            output_dir="./results/multigpu",          # Output directory
            num_train_epochs=1,              # Total number of training epochs
            per_device_train_batch_size=8,   # Batch size per device during training
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[hfgplo]
        )

        trainer.train()

if __name__ == '__main__':
    unittest.main(exit=True)
