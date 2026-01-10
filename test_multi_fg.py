#!/usr/bin/env python3
"""
Test script for multi-foreground subject support in CharaConsist
"""

import sys
import os
sys.path.append('.')

def test_prompt_parsing():
    """Test multi-subject prompt parsing"""
    from inference import parse_prompt_with_multi_fg, modify_prompt_and_get_lengths

    print("=" * 50)
    print("Testing Multi-Subject Prompt Parsing")
    print("=" * 50)

    # Test prompt
    test_prompt = 'in a park#a girl with brown hair#a boy with blue shirt#playing together'

    try:
        bg, fg_parts, act = parse_prompt_with_multi_fg(test_prompt)
        print(f"✓ Parsed prompt successfully:")
        print(f"  Background: {bg}")
        print(f"  Foreground parts: {fg_parts}")
        print(f"  Action: {act}")

        # Mock token length function for testing
        def mock_get_text_tokens_length(pipe, text):
            # Simple mock: return length based on word count
            return len(text.split()) * 2  # Approximate token count

        # Temporarily replace the function
        import inference
        original_func = inference.get_text_tokens_length
        inference.get_text_tokens_length = mock_get_text_tokens_length

        try:
            prompt, bg_len, fg_lengths, real_len = modify_prompt_and_get_lengths(bg, fg_parts, act, None)
            print(f"✓ Generated combined prompt: {prompt}")
            print(f"  Background length: {bg_len}")
            print(f"  Foreground lengths: {fg_lengths}")
            print(f"  Real length: {real_len}")
        finally:
            inference.get_text_tokens_length = original_func
        print(f"✓ Generated combined prompt: {prompt}")
        print(f"  Background length: {bg_len}")
        print(f"  Foreground lengths: {fg_lengths}")
        print(f"  Real length: {real_len}")

        return True
    except Exception as e:
        print(f"✗ Prompt parsing failed: {e}")
        return False

def test_load_prompt_file():
    """Test loading prompts file with multi-subject support"""
    from inference import load_prompt_file

    print("\n" + "=" * 50)
    print("Testing Multi-Subject Prompt File Loading")
    print("=" * 50)

    # Create test prompts file
    test_content = """in a park#a girl with brown hair#a boy with blue shirt#playing together
in a garden#a woman with red dress#a man with suit#walking together
"""

    test_file = '/tmp/test_multi_fg_prompts.txt'
    with open(test_file, 'w') as f:
        f.write(test_content)

    try:
        # Mock pipe object
        class MockPipe:
            def __init__(self):
                pass

        mock_pipe = MockPipe()

        # Mock the text length functions
        import inference
        original_get_text_tokens_length = inference.get_text_tokens_length

        def mock_get_text_tokens_length(pipe, text):
            return len(text.split()) * 2

        inference.get_text_tokens_length = mock_get_text_tokens_length

        all_prompt_info = load_prompt_file(mock_pipe, test_file)

        # Restore original function
        inference.get_text_tokens_length = original_get_text_tokens_length

        print(f"✓ Loaded {len(all_prompt_info)} prompt groups")
        for i, (prompts, bg_lens, fg_lengths_list, real_lens) in enumerate(all_prompt_info):
            print(f"  Group {i}: {len(prompts)} prompts")
            for j, (prompt, bg_len, fg_lengths, real_len) in enumerate(zip(prompts, bg_lens, fg_lengths_list, real_lens)):
                print(f"    Prompt {j}: fg_lengths={fg_lengths}")

        return True
    except Exception as e:
        print(f"✗ Prompt file loading failed: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_attention_processor_functions():
    """Test attention processor multi-fg functions"""
    print("\n" + "=" * 50)
    print("Testing Attention Processor Multi-FG Functions")
    print("=" * 50)

    try:
        from models.attention_processor_characonsist import CharaConsistAttnProcessor2_0

        # Test processor initialization
        processor = CharaConsistAttnProcessor2_0(size=(64, 64))
        print("✓ CharaConsistAttnProcessor2_0 initialized successfully")

        # Test fg_lengths attribute
        processor.fg_lengths = [10, 15]  # Two foreground subjects
        print(f"✓ Set fg_lengths: {processor.fg_lengths}")

        return True
    except Exception as e:
        print(f"✗ Attention processor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("CharaConsist Multi-Foreground Subject Test Suite")
    print("=" * 60)

    tests = [
        test_prompt_parsing,
        test_load_prompt_file,
        test_attention_processor_functions,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Multi-foreground support is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
