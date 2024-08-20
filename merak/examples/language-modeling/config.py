from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    BertConfig,
    BertForMaskedLM,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.configuration_utils import PretrainedConfig

from bloom.configuration_bloom import BloomConfig
from bloom.modeling_bloom import BloomForCausalLM


def load_config_and_model(model_name: str):
    if model_name.startswith("gpt3"):
        return load_gpt3(model_name)
    if model_name.startswith("bert"):
        return load_bert(model_name)
    if model_name.startswith("t5"):
        return load_t5(model_name)
    if model_name.startswith("bloom"):
        return load_bloom(model_name)
    raise NotImplementedError(f"Unknown model '{model_name}'")


def load_bloom_config(model_name: str):
    if model_name in ["bloom-560m", "bloom-1b1", "bloom-1b7", "bloom-3b", "bloom-7b1", "bloom"]:
        config = BloomConfig.from_pretrained(f"bigscience/{model_name}")
    else:
        config = BloomConfig.from_pretrained(f"bigscience/bloom")
        assert isinstance(config, BloomConfig)
        if model_name == "bloom-16b":  # 16387.19 M
            config.n_layer = 48
            config.hidden_size = 5120
            config.n_head = 32
        elif model_name == "bloom-42b":  # 41625.45 M
            config.n_layer = 64
            config.hidden_size = 7200
            config.n_head = 48
        elif model_name == "bloom-73b":  # 73195.32 M
            config.n_layer = 64
            config.hidden_size = 9600
            config.n_head = 64

    config.use_cache = False
    config.return_dict = False
    return config

def load_bloom(model_name: str):
    config = load_bloom_config(model_name)
    assert isinstance(config, BloomConfig)
    model = BloomForCausalLM(config)
    print(f"Loaded '{model_name}' with {count_parameters(model)/1e6:.2f}M parameters")

    return config, model


def load_gpt3(model_name: str):
    config = load_gpt3_config(model_name)
    return load_gpt3_model(model_name, config)


def load_gpt3_config(model_name: str):
    config = GPT2Config.from_pretrained("gpt2")
    config.use_cache = False
    config.return_dict = False

    if model_name == "gpt3-small":
        config.n_embd = 768
        config.n_layer = 12
        config.n_head = 12
    elif model_name == "gpt3-medium":
        config.n_embd = 1024
        config.n_layer = 24
        config.n_head = 16
    elif model_name == "gpt3-large":
        config.n_embd = 1536
        config.n_layer = 24
        config.n_head = 16
    elif model_name == "gpt3-xl":
        config.n_embd = 2048
        config.n_layer = 24
        config.n_head = 16
    elif model_name == "gpt3-2.7b":
        config.n_embd = 2560
        config.n_layer = 32
        config.n_head = 32
    elif model_name == "gpt3-6.7b":
        config.n_embd = 4096
        config.n_layer = 32
        config.n_head = 32
    elif model_name == "gpt3-13b":
        config.n_embd = 5120
        config.n_layer = 40
        config.n_head = 40
    elif model_name == "gpt3-39b":  # 39491.63 M
        config.n_embd = 8192
        config.n_layer = 48
        config.n_head = 64
    elif model_name == "gpt3-80b":  # 80600.16 M
        config.n_embd = 9600
        config.n_layer = 72
        config.n_head = 64
    elif model_name == "gpt3-175b":  # 175196.63 M
        config.n_embd = 12288
        config.n_layer = 96
        config.n_head = 96
    else:
        raise NotImplementedError(
            f"'{model_name}' is not a supported GPT3 variant.",
        )

    return config


def load_gpt3_model(model_name: str, config: PretrainedConfig):
    model = GPT2LMHeadModel(config)
    print(f"Loaded '{model_name}' with {count_parameters(model)/1e6:.2f}M parameters")

    return config, model


def load_bert_config(model_name: str):
    if model_name == "bert-huge-uncased":  # 1276.40 M
        config = BertConfig.from_pretrained("bert-large-uncased")
        config.hidden_size = 2048
        config.intermediate_size = 2048 * 4
    else:
        config = BertConfig.from_pretrained(model_name)
    config.return_dict = False
    return config

def load_bert(model_name: str):
    config = load_bert_config(model_name)
    model = BertForMaskedLM(config)
    print(f"Loaded '{model_name}' with {count_parameters(model)/1e6:.2f}M parameters")

    return config, model


def load_t5_config(model_name: str):
    config = T5Config.from_pretrained(model_name)
    config.use_cache = False
    config.return_dict = False

    return config

def load_t5(model_name: str):
    config = load_t5_config(model_name)
    model = T5ForConditionalGeneration(config)
    print(f"Loaded '{model_name}' with {count_parameters(model)/1e6:.2f}M parameters")

    return config, model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
