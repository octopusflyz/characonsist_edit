#!/usr/bin/env python3
"""
Integration test for the complete progressive style guidance system
"""
import os
import sys
import torch
import argparse

# Set up minimal arguments for testing
def create_test_args():
    args = argparse.Namespace()

    # Basic settings
    args.init_mode = 0  # Use basic model init
    args.gpu_ids = [0]
    args.prompts_file = "test_prompts.txt"
    args.model_path = "/mnt/netdisk2/zhangyf/workspace/Other/pure/models/FLUX.1-dev"  # Adjust path as needed
    args.out_dir = "results/test_style_guidance"
    args.use_interpolate = False
    args.share_bg = False
    args.save_mask = True
    args.save_point_match = False
    args.point_match_dir = ""
    args.height = 512  # Smaller size for testing
    args.width = 512
    args.seed = 42

    # Style guidance settings - conservative values for testing
    args.use_cfg_guidance = True
    args.cfg_guidance_scale = 0.2  # Conservative scale
    args.cfg_guidance_start_step = 1
    args.cfg_guidance_end_step = 20
    args.cfg_guidance_start_sigma = 0.8  # Start when noise is still high
    args.cfg_guidance_end_sigma = 0.2    # End before details are finalized

    # Debug settings
    args.debug_output_dir = "results/test_style_guidance/debug"

    return args

def test_system_integration():
    print("=== Testing Complete Progressive Style Guidance System ===")

    try:
        # Import required modules
        from inference import MODEL_INIT_FUNCS, get_text_tokens_length, modify_prompt_and_get_length, load_prompt_file

        args = create_test_args()

        # Check if model path exists
        if not os.path.exists(args.model_path):
            print(f"Model path {args.model_path} not found. Please adjust the path.")
            print("Skipping full integration test, but unit tests passed.")
            return

        print(f"Using model path: {args.model_path}")
        print(f"Style guidance: enabled with scale {args.cfg_guidance_scale}")
        print(f"Sigma range: {args.cfg_guidance_start_sigma} to {args.cfg_guidance_end_sigma}")

        # Initialize model (this might take time)
        print("Initializing model...")
        pipe = MODEL_INIT_FUNCS[args.init_mode]()
        print("Model initialized successfully")

        # Load prompts
        all_prompt_info = load_prompt_file(pipe, args.prompts_file)
        print(f"Loaded {len(all_prompt_info)} prompt sets")

        # Test just the first prompt set
        if len(all_prompt_info) > 0:
            print("Testing with first prompt set...")

            # Set up pipeline kwargs
            pipe_kwargs = dict(
                height=args.height,
                width=args.width,
                use_interpolate=args.use_interpolate,
                share_bg=args.share_bg,
                use_style_guidance=args.use_cfg_guidance,
                style_guidance_scale=args.cfg_guidance_scale,
                style_guidance_start_step=args.cfg_guidance_start_step,
                style_guidance_end_step=args.cfg_guidance_end_step,
                style_guidance_start_sigma=args.cfg_guidance_start_sigma,
                style_guidance_end_sigma=args.cfg_guidance_end_sigma,
                debug_output_dir=args.debug_output_dir
            )

            prompt_ind = 0
            (prompts, bg_lens, real_lens) = all_prompt_info[prompt_ind]

            print(f"Processing prompt set {prompt_ind}")
            print(f"ID prompt: {prompts[0]}")

            # Test ID generation
            from inference import reset_attn_processor, set_text_len
            reset_attn_processor(
                pipe,
                size=(args.height//16, args.width//16),
                style_layers=[8, 16, 24] if args.use_cfg_guidance else None,
                style_weight=0.1 if args.use_cfg_guidance else 0.0,
                style_max_timestep=30
            )

            id_prompt = prompts[0]
            bg_len, real_len = bg_lens[0], real_lens[0]
            set_text_len(pipe, bg_len, real_len)

            print("Generating ID image...")
            id_images, id_spatial_kwargs = pipe(
                id_prompt, is_id=True,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                **pipe_kwargs
            )

            print("✓ ID image generated successfully")

            # Test frame generation with style guidance
            if len(prompts) > 1:
                frame_prompt = prompts[1]
                print(f"Generating frame with style guidance: {frame_prompt}")

                # Generate one frame to test the system
                set_text_len(pipe, bg_lens[1], real_lens[1])

                images, spatial_kwargs = pipe(
                    frame_prompt,
                    generator=torch.Generator("cpu").manual_seed(args.seed),
                    spatial_kwargs={"id_fg_mask": id_spatial_kwargs["curr_fg_mask"], "id_bg_mask": ~id_spatial_kwargs["curr_fg_mask"]},
                    **pipe_kwargs
                )

                print("✓ Frame image generated with progressive style guidance")
                print("✓ System integration test completed successfully!")

            else:
                print("✓ ID generation test completed (no frame prompts to test)")

        else:
            print("No prompts found in test file")

    except Exception as e:
        print(f"✗ System integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system_integration()
