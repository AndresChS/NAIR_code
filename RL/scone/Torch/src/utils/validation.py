"""
    Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
    source	:: https://github.com/AndresChS/NAIR_Code
"""

from tensordict import TensorDict
import torch

# -------------------------------
# Validation of batch
# -------------------------------
def validate_batch(batch, required_keys=None):
    if required_keys is None:
        required_keys = ["observation", "reward", "action", "done", "advantage", "state_value", "value_target"]

    print("\n" + "="*10 + " VALIDANDO TENSORDICT " + "="*10)
    errors = False

    # Check if batch is empty
    if batch.batch_size == torch.Size([]) or len(batch.batch_size) == 0:
        print("[ERROR] Batch está vacío (batch_size=[])")
        errors = True

    # Check required keys
    for key in required_keys:
        if key not in batch:
            print(f"[ERROR] Falta la clave '{key}' en el batch.")
            errors = True
            continue

        val = batch.get(key)
        if val is None:
            print(f"[ERROR] batch['{key}'] es None.")
            errors = True
        elif not isinstance(val, torch.Tensor):
            print(f"[ERROR] batch['{key}'] no es un tensor.")
            errors = True
        elif torch.isnan(val).any():
            print(f"[ERROR] batch['{key}'] contiene NaN.")
            errors = True
        elif torch.isinf(val).any():
            print(f"[ERROR] batch['{key}'] contiene Inf.")
            errors = True
        else:
            print(f"[OK] '{key}' shape: {val.shape}, min: {val.min().item():.4f}, max: {val.max().item():.4f}")

    # Check nested 'next' tensordict
    next_td = batch.get("next", None)
    if next_td is not None:
        if isinstance(next_td, TensorDict):
            if "observation" in next_td:
                next_obs = next_td.get("observation")
                if next_obs is None or not isinstance(next_obs, torch.Tensor):
                    print("[ERROR] batch['next']['observation'] es inválido.")
                    errors = True
                else:
                    print(f"[OK] 'next.observation' shape: {next_obs.shape}")
            else:
                print("[ERROR] Falta 'observation' en batch['next'].")
                errors = True
        else:
            print("[ERROR] batch['next'] no es un TensorDict.")
            errors = True
    else:
        print("[ERROR] batch no contiene la clave 'next'.")
        errors = True

    print("="*10 + " VALIDACIÓN COMPLETA " + "="*10 + "\n")
    return not errors