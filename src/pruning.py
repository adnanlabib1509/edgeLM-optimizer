import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """
    Prune the model by removing a percentage of the smallest weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model