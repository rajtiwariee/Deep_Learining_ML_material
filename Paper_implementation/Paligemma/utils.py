from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
from transformers import AutoModel, AutoConfig
from transformers import PaliGemmaForConditionalGeneration as palimodel
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
login(token=os.environ.get("HUGGING_FACE_TOKEN"))

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    #This will tie the weights of the linear layer with the weights of the embedding layer
    model.tie_weights()

    return (model, tokenizer)


def load_hf_model_weights(model_name: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]: 
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    assert tokenizer.padding_side == "right"
    
    # Load the model
    hf_model = palimodel.from_pretrained(model_name).to(device)
    #save the weights
    # Get Hugging Face state dict (weights)
    hf_state_dict = hf_model.state_dict()

    # Load the model's config
    with open("config.json", "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)
    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(hf_state_dict, strict=False)

    # Tie weights
    #This will tie the weights of the linear layer with the weights of the embedding layer
    model.tie_weights()

    return (model, tokenizer)