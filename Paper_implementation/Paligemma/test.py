from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from modeling_gemma import PaliGemmaConfig
import json

# def test_class_call():
#     # Load the model's config
#     with open("config.json", "r") as f:
#         model_config_file = json.load(f)
#         config = PaliGemmaConfig(**model_config_file)
#     # Create the model using the configuration
#     model = PaliGemmaForConditionalGeneration(config)
    
#     return model

if __name__ == "__main__":
    height = [0,0,0,0] + [int(i) for i in ["1", "0", "1", "0"]]
    print(height)