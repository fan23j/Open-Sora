def calculate_weight_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def push_to_device(item, device, dtype):
    if isinstance(item, dict):
        return {k: push_to_device(v, device, dtype) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return type(item)(push_to_device(v, device, dtype) for v in item)
    elif hasattr(item, "to"):
        return item.to(device, dtype)
    else:
        return item
