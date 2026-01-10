#!/usr/bin/env python3
"""
Test script for frame generation in multi-foreground mode
"""

import sys
import os
sys.path.append('.')

def test_frame_generation_logic():
    """Test the frame generation logic without running full inference"""
    print("=" * 50)
    print("Testing Frame Generation Logic")
    print("=" * 50)

    # Create a simple test prompt file
    test_content = """in a park#a girl with brown hair#a boy with blue shirt#playing together
in a garden#a girl with brown hair#a boy with blue shirt#walking together
"""

    test_file = '/tmp/test_frame_prompts.txt'
    with open(test_file, 'w') as f:
        f.write(test_content)

    try:
        # Mock pipe object
        class MockPipe:
            def __init__(self):
                self.fg_lengths = None

        mock_pipe = MockPipe()

        # Mock the text length functions
        import inference
        original_get_text_tokens_length = inference.get_text_tokens_length

        def mock_get_text_tokens_length(pipe, text):
            return len(text.split()) * 2

        inference.get_text_tokens_length = mock_get_text_tokens_length

        # Test load_prompt_file
        all_prompt_info = inference.load_prompt_file(mock_pipe, test_file)

        print(f"✓ Loaded {len(all_prompt_info)} prompt groups")

        for prompt_ind, (prompts, bg_lens, fg_lengths_list, real_lens) in enumerate(all_prompt_info):
            print(f"\nGroup {prompt_ind}:")
            print(f"  Total prompts: {len(prompts)}")
            print(f"  ID prompt: {prompts[0][:50]}...")
            print(f"  ID fg_lengths: {fg_lengths_list[0]}")

            frm_prompts = prompts[1:]
            print(f"  Frame prompts: {len(frm_prompts)}")

            for i, (prompt, fg_lengths) in enumerate(zip(frm_prompts, fg_lengths_list[1:])):
                print(f"    Frame {i}: fg_lengths={fg_lengths}")
                print(f"      Prompt: {prompt[:50]}...")

        # Restore original function
        inference.get_text_tokens_length = original_get_text_tokens_length

        # Check if frames would be generated
        if len(all_prompt_info) > 0:
            prompts, bg_lens, fg_lengths_list, real_lens = all_prompt_info[0]
            frm_prompts = prompts[1:]

            if len(frm_prompts) > 0:
                print(f"\n✓ Frame generation would proceed with {len(frm_prompts)} frames")
                return True
            else:
                print(f"\n✗ No frame prompts found - this is the bug!")
                return False

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    success = test_frame_generation_logic()
    if success:
        print("\n🎉 Frame generation logic test PASSED")
    else:
        print("\n❌ Frame generation logic test FAILED")
    sys.exit(0 if success else 1)
