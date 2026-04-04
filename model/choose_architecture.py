import torch


# Wybiera odpowiednią architektruę urządenia do trenowania modelu
# Jeśli urządzenie posiada kartę graficzną CUDA,
# Jeśli macbook z procesorem Apple Silicon to mps
# Jeśli nie ma żadnego urządzenia to cpu
def choose_architecture():
    use_cuda = torch.cuda.is_available()

    use_mps = (
        torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    )

    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device
