#!/usr/bin/env python3
import numpy as np

def get_style_guidance_weight(weight, start_step, end_step):
    print(f"[CFG_WEIGHT_CALC] Input: weight={weight}, start_step={start_step}, end_step={end_step}")
    if weight <= 0:
        print("[CFG_WEIGHT_CALC] Weight <= 0, returning empty dict")
        return {}
    steps = np.arange(start_step, end_step)
    print(f"[CFG_WEIGHT_CALC] Steps: {steps}")
    weights = weight * (1 - (steps - start_step) / (end_step - start_step))
    weights = np.maximum(weights, 0)
    print(f"[CFG_WEIGHT_CALC] Weights: {weights}")
    weight_dict = dict(zip(steps, weights.tolist()))
    print(f"[CFG_WEIGHT_CALC] Final dict: {weight_dict}")
    return weight_dict

if __name__ == "__main__":
    print("Testing CFG weight calculation...")
    weights = get_style_guidance_weight(100.0, 40, 50)
    print(f"Final result: {weights}")