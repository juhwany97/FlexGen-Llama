"""
The OPT model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "meta-llama-3-8b"
    num_hidden_layers: int = 32
    max_seq_len: int = 8192
    hidden_size: int = 4096
    n_head: int = 32
    n_kv_head: int = 8
    input_dim: int = 4096
    ffn_embed_dim: int = 14336
    pad: int = 1
    activation_fn: str = 'silu'
    vocab_size: int = 128256
    rope_theta: int = 500000
    layer_norm_eps: float = 0.00001
    pad_token_id: int = 1
    dtype: type = np.float16

    def model_bytes(self):
        h = self.input_dim
        n_q = self.n_head
        n_kv = self.n_kv_head
        w_kv_dim = h / (n_q // n_kv)
        ffn_dim = self.ffn_embed_dim
        
        return 	2 * (self.num_hidden_layers * (
        # self-attention (GQA)
        h * (h + 2 * w_kv_dim) + h * h +
        # mlp (3-way)
        3 * h * ffn_dim +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * h)


    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * (self.input_dim * self.n_head) // self.n_kv_head * self.dtype.nbytes 

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * self.dtype.nbytes


def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    arch_name = name

    if arch_name == "meta-llama-3-8b":
        config = LlamaConfig(name=name,
            num_hidden_layers=32, max_seq_len=8192, hidden_size=4096, n_head=32, n_kv_head=8,
            input_dim=4096, ffn_embed_dim=14336,
        )
    elif arch_name == "meta-llama-3-70b":
        config = LlamaConfig(name=name,
            num_hidden_layers=80, max_seq_len=8192, hidden_size=8192, n_head=64, n_kv_head=8,
            input_dim=8192, ffn_embed_dim=28672,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def download_opt_weights_old(model_name, path):
    """Download weights from huggingface."""
    import torch
    from transformers import OPTForCausalLM, BloomForCausalLM, LlamaForCausalLM

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        hf_model_name = "bigscience/" + model_name
        model_class = BloomForCausalLM
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "llama" in model_name:
        hf_model_name = "meta-llama" + model_name
    else:
        raise ValueError("Invalid model name: {model_name}")

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16,
                                        _fast_init=True)
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    print(f"Convert the weights to numpy format under {path} ...")
    if "opt" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "galactica" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "bloom" in model_name:
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "llama" in model_name:
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    else:
        raise ValueError("Invalid model name: {model_name}")


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_opt_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.opt.modeling_opt.OPTPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


def download_opt_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="~/opt_weights")
    args = parser.parse_args()

    download_opt_weights(args.model, args.path)
