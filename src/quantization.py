import torch
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from awq import AutoAWQForCausalLM

def quantize_model(model, quantization_type="dynamic", model_name="gpt2"):
    """
    Quantize the model using various techniques.
    """
    if quantization_type == "dynamic":
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif quantization_type == "static":
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        return model
    elif quantization_type == "gptq":
        # GPTQ quantization
        quantize_config = BaseQuantizeConfig(
            bits=4,  # quantize model to 4-bit
            group_size=128,  # group size
            desc_act=False,  # disable activation description
        )
        model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
        return model
    elif quantization_type == "awq":
        # AWQ quantization
        model = AutoAWQForCausalLM.from_pretrained(model_name, 
                                                   bits=4,  # quantize to 4-bit
                                                   group_size=128)
        return model
    else:
        raise ValueError("Invalid quantization type. Choose 'dynamic', 'static', 'gptq', or 'awq'.")