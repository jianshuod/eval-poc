#!/usr/bin/env python3
"""
Token Counting Verification Test for Safety Confirmation V6

This test verifies that the token counting implementation in safety confirmation v6
works correctly when using the aihubmix.com/v1 proxy service. The proxy service may
have different response formats than the standard OpenAI API, which could break the
assumptions made in the extract_usage() function.

Usage:
    python test_token_counting.py [--model MODEL] [--verbose] [--output OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "safety_confirmation" / "src"))

try:
    from openai import OpenAI
    from safety_confirmation.utils import extract_usage
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure safety_confirmation is installed: pip install -e ../safety_confirmation")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# Models from grid-search-multi-model-test-agentdojo.yaml
DEFAULT_MODELS = [
    "sophnet-glm-4.7",
    "alicloud-deepseek-v3.2",
    "gemini-3-flash-preview",
    "qwen3-235b-a22b-thinking-2507",
    "alicloud-qwen3.5-35b-a3b",
    "doubao-seed-2-0-pro",
]

# API Configuration
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1")

# Test prompts
SHORT_PROMPT = "Hello, how are you?"
LONG_SYSTEM_PROMPT = "You are a helpful assistant. " * 100  # ~2000 tokens
REASONING_PROMPT = "Solve this step by step: What is 15 * 23 - 47?"


# =============================================================================
# Test Results Data Classes
# =============================================================================

@dataclass
class ModelTestResult:
    """Test results for a single model."""
    model: str
    success: bool
    error_message: str = ""
    raw_response_fields: list[str] = field(default_factory=list)
    usage_fields: list[str] = field(default_factory=list)
    extract_usage_result: dict = field(default_factory=dict)
    basic_tokens_test: dict = field(default_factory=dict)
    cache_behavior_test: dict = field(default_factory=dict)
    reasoning_tokens_test: dict = field(default_factory=dict)
    image_audio_tokens_test: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Test Functions
# =============================================================================

def test_raw_response_structure(model: str, client: OpenAI) -> dict:
    """
    Test 1: Inspect raw response object to identify all available fields.

    Returns:
        Dict with 'fields' list and 'sample_response' dict
    """
    print(f"  Testing raw response structure...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SHORT_PROMPT}],
            max_tokens=50,
        )

        # Collect all attributes from response and usage
        fields = []

        # Response-level fields
        if hasattr(response, 'id'):
            fields.append('id')
        if hasattr(response, 'object'):
            fields.append('object')
        if hasattr(response, 'created'):
            fields.append('created')
        if hasattr(response, 'model'):
            fields.append('model')
        if hasattr(response, 'choices'):
            fields.append('choices')
        if hasattr(response, 'usage'):
            fields.append('usage')

        # Usage-level fields
        usage_fields = []
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            for attr in dir(usage):
                if not attr.startswith('_'):
                    usage_fields.append(attr)

            # Check for nested details
            if hasattr(usage, 'prompt_tokens_details'):
                usage_fields.append('prompt_tokens_details')
                if usage.prompt_tokens_details:
                    for attr in dir(usage.prompt_tokens_details):
                        if not attr.startswith('_'):
                            usage_fields.append(f'prompt_tokens_details.{attr}')

            if hasattr(usage, 'completion_tokens_details'):
                usage_fields.append('completion_tokens_details')
                if usage.completion_tokens_details:
                    for attr in dir(usage.completion_tokens_details):
                        if not attr.startswith('_'):
                            usage_fields.append(f'completion_tokens_details.{attr}')

        return {
            'fields': fields,
            'usage_fields': usage_fields,
            'sample_response': str(response)[:500],  # Truncated for report
        }

    except Exception as e:
        return {
            'fields': [],
            'usage_fields': [],
            'error': str(e),
        }


def test_extract_usage(model: str, client: OpenAI, response) -> dict:
    """
    Test 2: Validate extract_usage() function output.

    Returns:
        Dict with extract_usage result and validation
    """
    print(f"  Testing extract_usage() function...")

    try:
        usage_dict = extract_usage(response)

        # Validate required fields
        required_fields = ['input_tokens', 'output_tokens', 'total_tokens']
        optional_fields = [
            'cache_read', 'cache_write', 'reasoning',
            'image_tokens_input', 'image_tokens_output',
            'audio_tokens_input', 'audio_tokens_output'
        ]

        validation = {
            'usage_dict': usage_dict,
            'has_required': all(field in usage_dict for field in required_fields),
            'missing_required': [f for f in required_fields if f not in usage_dict],
            'present_optional': [f for f in optional_fields if usage_dict.get(f) is not None],
        }

        return validation

    except Exception as e:
        return {
            'error': str(e),
            'usage_dict': {},
        }


def test_basic_tokens(model: str, client: OpenAI) -> dict:
    """
    Test 3: Verify input/output token accuracy.

    Returns:
        Dict with token counts and validation
    """
    print(f"  Testing basic token counting...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello world' in response."}
            ],
            max_tokens=20,
        )

        usage_dict = extract_usage(response)

        result = {
            'usage_dict': usage_dict,
            'input_tokens': usage_dict.get('input_tokens'),
            'output_tokens': usage_dict.get('output_tokens'),
            'total_tokens': usage_dict.get('total_tokens'),
            'total_matches_sum': (
                usage_dict.get('total_tokens') ==
                usage_dict.get('input_tokens') + usage_dict.get('output_tokens')
                if usage_dict.get('total_tokens') and usage_dict.get('input_tokens') and usage_dict.get('output_tokens')
                else None
            ),
        }

        return result

    except Exception as e:
        return {
            'error': str(e),
        }


def test_cache_behavior(model: str, client: OpenAI) -> dict:
    """
    Test 4: Sequential requests to detect caching.

    Returns:
        Dict with cache behavior analysis
    """
    print(f"  Testing cache behavior...")

    try:
        # Request 1: Establish cache with long system prompt
        response1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LONG_SYSTEM_PROMPT},
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=20,
        )
        usage1 = extract_usage(response1)

        time.sleep(1)  # Brief delay

        # Request 2: Same system prompt (should show cache_read if supported)
        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LONG_SYSTEM_PROMPT},
                {"role": "user", "content": "What is 3+3?"}
            ],
            max_tokens=20,
        )
        usage2 = extract_usage(response2)

        time.sleep(1)  # Brief delay

        # Request 3: Different system prompt (control)
        response3 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a different assistant. " * 100},
                {"role": "user", "content": "What is 4+4?"}
            ],
            max_tokens=20,
        )
        usage3 = extract_usage(response3)

        result = {
            'request1_usage': usage1,
            'request2_usage': usage2,
            'request3_usage': usage3,
            'cache_read_detected': (
                usage2.get('cache_read') is not None and usage2.get('cache_read') > 0
            ),
            'cache_write_detected': (
                usage1.get('cache_write') is not None and usage1.get('cache_write') > 0
            ),
            'cache_read_value': usage2.get('cache_read'),
            'cache_write_value': usage1.get('cache_write'),
        }

        return result

    except Exception as e:
        return {
            'error': str(e),
        }


def test_reasoning_tokens(model: str, client: OpenAI) -> dict:
    """
    Test 5: Specific test for thinking/reasoning models.

    Returns:
        Dict with reasoning token analysis
    """
    print(f"  Testing reasoning tokens...")

    try:
        # Use a prompt that might trigger reasoning
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": REASONING_PROMPT}
            ],
            max_tokens=200,
        )

        usage_dict = extract_usage(response)

        result = {
            'usage_dict': usage_dict,
            'has_reasoning_tokens': usage_dict.get('reasoning') is not None,
            'reasoning_token_count': usage_dict.get('reasoning'),
            'is_thinking_model': 'thinking' in model.lower(),
        }

        return result

    except Exception as e:
        return {
            'error': str(e),
        }


def test_image_audio_tokens(model: str, client: OpenAI) -> dict:
    """
    Test 6: Test for multimedia token support.

    Returns:
        Dict with image/audio token analysis
    """
    print(f"  Testing image/audio token fields...")

    try:
        # Make a basic request to check response structure
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=20,
        )

        usage_dict = extract_usage(response)

        result = {
            'usage_dict': usage_dict,
            'has_image_input_tokens': usage_dict.get('image_tokens_input') is not None,
            'has_image_output_tokens': usage_dict.get('image_tokens_output') is not None,
            'has_audio_input_tokens': usage_dict.get('audio_tokens_input') is not None,
            'has_audio_output_tokens': usage_dict.get('audio_tokens_output') is not None,
        }

        return result

    except Exception as e:
        return {
            'error': str(e),
        }


def test_safety_confirmation_flow(model: str, client: OpenAI) -> dict:
    """
    Test 7: Full safety confirmation v6 flow test.

    Returns:
        Dict with safety confirmation flow results
    """
    print(f"  Testing safety confirmation flow...")

    try:
        # This is a basic test - full flow would require safety_confirmation package
        # We're testing if the model can handle tool calls correctly
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            tools=[],
            max_tokens=50,
        )

        usage_dict = extract_usage(response)

        result = {
            'usage_dict': usage_dict,
            'has_choices': hasattr(response, 'choices') and len(response.choices) > 0,
            'has_message': (
                hasattr(response, 'choices') and
                len(response.choices) > 0 and
                hasattr(response.choices[0], 'message')
            ),
            'response_content': (
                response.choices[0].message.content[:100]
                if hasattr(response, 'choices') and len(response.choices) > 0
                else None
            ),
        }

        return result

    except Exception as e:
        return {
            'error': str(e),
        }


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(results: list[ModelTestResult], base_url: str, output_path: Optional[str] = None) -> str:
    """Generate a comprehensive markdown report from test results."""

    report_lines = [
        "# Token Counting Verification Test Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**API Endpoint**: {base_url}",
        f"**Models Tested**: {len(results)}",
        "",
        "## Test Summary",
        "",
        "| Model | Status | Issues | Warnings |",
        "|-------|--------|--------|----------|",
    ]

    for result in results:
        status = "✅ PASS" if result.success else "❌ FAIL"
        report_lines.append(
            f"| {result.model} | {status} | {len(result.issues)} | {len(result.warnings)} |"
        )

    report_lines.extend([
        "",
        "## Per-Model Results",
        "",
    ])

    for result in results:
        report_lines.extend([
            f"### {result.model}",
            "",
            f"**Status**: {'✅ PASS' if result.success else '❌ FAIL'}",
            "",
        ])

        if result.error_message:
            report_lines.extend([
                f"**Error**: {result.error_message}",
                "",
            ])

        if result.raw_response_fields:
            report_lines.extend([
                "#### Raw Response Structure",
                "",
                "**Response-level fields**:",
                f"```",
                ", ".join(result.raw_response_fields),
                "```",
                "",
                "**Usage-level fields**:",
                f"```",
                ", ".join(result.usage_fields),
                "```",
                "",
            ])

        if result.extract_usage_result:
            report_lines.extend([
                "#### extract_usage() Result",
                "",
                f"```json",
                json.dumps(result.extract_usage_result, indent=2),
                "```",
                "",
            ])

        if result.basic_tokens_test:
            report_lines.extend([
                "#### Basic Token Counting",
                "",
                f"- Input tokens: {result.basic_tokens_test.get('input_tokens', 'N/A')}",
                f"- Output tokens: {result.basic_tokens_test.get('output_tokens', 'N/A')}",
                f"- Total tokens: {result.basic_tokens_test.get('total_tokens', 'N/A')}",
                f"- Total matches sum: {result.basic_tokens_test.get('total_matches_sum', 'N/A')}",
                "",
            ])

        if result.cache_behavior_test:
            report_lines.extend([
                "#### Cache Behavior",
                "",
                f"- Cache read detected: {result.cache_behavior_test.get('cache_read_detected', False)}",
                f"- Cache write detected: {result.cache_behavior_test.get('cache_write_detected', False)}",
                f"- Cache read value: {result.cache_behavior_test.get('cache_read_value', 'N/A')}",
                f"- Cache write value: {result.cache_behavior_test.get('cache_write_value', 'N/A')}",
                "",
            ])

        if result.reasoning_tokens_test:
            report_lines.extend([
                "#### Reasoning Tokens",
                "",
                f"- Has reasoning tokens: {result.reasoning_tokens_test.get('has_reasoning_tokens', False)}",
                f"- Reasoning token count: {result.reasoning_tokens_test.get('reasoning_token_count', 'N/A')}",
                f"- Is thinking model: {result.reasoning_tokens_test.get('is_thinking_model', False)}",
                "",
            ])

        if result.image_audio_tokens_test:
            report_lines.extend([
                "#### Image/Audio Token Fields",
                "",
                f"- Image input tokens: {result.image_audio_tokens_test.get('has_image_input_tokens', False)}",
                f"- Image output tokens: {result.image_audio_tokens_test.get('has_image_output_tokens', False)}",
                f"- Audio input tokens: {result.image_audio_tokens_test.get('has_audio_input_tokens', False)}",
                f"- Audio output tokens: {result.image_audio_tokens_test.get('has_audio_output_tokens', False)}",
                "",
            ])

        if result.issues:
            report_lines.extend([
                "#### Issues",
                "",
            ])
            for issue in result.issues:
                report_lines.append(f"- ❌ {issue}")
            report_lines.append("")

        if result.warnings:
            report_lines.extend([
                "#### Warnings",
                "",
            ])
            for warning in result.warnings:
                report_lines.append(f"- ⚠️ {warning}")
            report_lines.append("")

    # Cross-model comparison
    report_lines.extend([
        "## Cross-Model Comparison",
        "",
        "### Token Counting Fields Available",
        "",
        "| Field | Models with Field |",
        "|-------|-------------------|",
    ])

    # Collect all unique fields across models
    all_fields = set()
    for result in results:
        all_fields.update(result.usage_fields)

    for field in sorted(all_fields):
        models_with_field = [r.model for r in results if field in r.usage_fields]
        report_lines.append(f"| {field} | {', '.join(models_with_field)} |")

    report_lines.extend([
        "",
        "### extract_usage() Optional Fields Detection",
        "",
        "| Optional Field | Models Detecting |",
        "|----------------|------------------|",
    ])

    # Add rows for each optional field
    for field in ['cache_read', 'cache_write', 'reasoning']:
        models_with_field = ', '.join([
            r.model for r in results
            if isinstance(r.extract_usage_result.get('present_optional'), list)
            and field in r.extract_usage_result.get('present_optional', [])
        ]) or 'None'
        report_lines.append(f"| {field} | {models_with_field} |")

    report_lines.extend([
        "",
        "## Recommendations",
        "",
    ])

    # Add recommendations based on findings
    needs_fixes = False
    for result in results:
        if result.issues:
            needs_fixes = True
            break

    if needs_fixes:
        report_lines.extend([
            "### Issues Found",
            "",
            "The following issues were detected that may require fixes to `extract_usage()`:",
            "",
        ])
        for result in results:
            for issue in result.issues:
                report_lines.append(f"- **{result.model}**: {issue}")
        report_lines.append("")
    else:
        report_lines.extend([
            "### No Critical Issues",
            "",
            "All tested models work correctly with the current `extract_usage()` implementation.",
            "",
        ])

    report_text = "\n".join(report_lines)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report_text)
        print(f"\nReport saved to: {output_path}")

    return report_text


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests(
    models: list[str],
    api_key: str,
    base_url: str,
    verbose: bool = False,
    output: Optional[str] = None,
) -> list[ModelTestResult]:
    """Run all tests for the specified models."""

    results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print(f"{'='*60}")

        result = ModelTestResult(model=model, success=False)
        client = OpenAI(api_key=api_key, base_url=base_url)

        try:
            # Test 1: Raw response structure
            raw_result = test_raw_response_structure(model, client)
            if 'error' in raw_result:
                result.error_message = raw_result['error']
                result.issues.append(f"Raw response test failed: {raw_result['error']}")
                results.append(result)
                continue

            result.raw_response_fields = raw_result['fields']
            result.usage_fields = raw_result['usage_fields']

            if verbose:
                print(f"    Response fields: {result.raw_response_fields}")
                print(f"    Usage fields: {result.usage_fields}")

            # Get a response for subsequent tests
            test_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": SHORT_PROMPT}],
                max_tokens=50,
            )

            # Test 2: extract_usage()
            extract_result = test_extract_usage(model, client, test_response)
            result.extract_usage_result = extract_result.get('usage_dict', {})

            if not extract_result.get('has_required'):
                result.issues.append(
                    f"Missing required fields: {extract_result.get('missing_required', [])}"
                )

            if verbose:
                print(f"    extract_usage result: {result.extract_usage_result}")

            # Test 3: Basic tokens
            basic_result = test_basic_tokens(model, client)
            result.basic_tokens_test = basic_result

            if 'error' in basic_result:
                result.issues.append(f"Basic tokens test failed: {basic_result['error']}")
            elif not basic_result.get('total_matches_sum'):
                result.warnings.append(
                    f"Total tokens ({basic_result.get('total_tokens')}) != "
                    f"input ({basic_result.get('input_tokens')}) + output ({basic_result.get('output_tokens')})"
                )

            if verbose:
                print(f"    Basic tokens: input={basic_result.get('input_tokens')}, "
                      f"output={basic_result.get('output_tokens')}, "
                      f"total={basic_result.get('total_tokens')}")

            # Test 4: Cache behavior
            cache_result = test_cache_behavior(model, client)
            result.cache_behavior_test = cache_result

            if 'error' in cache_result:
                result.warnings.append(f"Cache behavior test failed: {cache_result['error']}")

            if verbose:
                print(f"    Cache read detected: {cache_result.get('cache_read_detected')}")
                print(f"    Cache write detected: {cache_result.get('cache_write_detected')}")

            # Test 5: Reasoning tokens
            reasoning_result = test_reasoning_tokens(model, client)
            result.reasoning_tokens_test = reasoning_result

            if 'error' in reasoning_result:
                result.warnings.append(f"Reasoning tokens test failed: {reasoning_result['error']}")

            if verbose:
                print(f"    Reasoning tokens: {reasoning_result.get('reasoning_token_count')}")

            # Test 6: Image/audio tokens
            multimedia_result = test_image_audio_tokens(model, client)
            result.image_audio_tokens_test = multimedia_result

            if 'error' in multimedia_result:
                result.warnings.append(f"Image/audio tokens test failed: {multimedia_result['error']}")

            # Test 7: Safety confirmation flow
            flow_result = test_safety_confirmation_flow(model, client)

            if 'error' in flow_result:
                result.warnings.append(f"Safety confirmation flow test failed: {flow_result['error']}")

            # Mark as successful if no critical issues
            result.success = len(result.issues) == 0

            print(f"\n  Result: {'✅ PASS' if result.success else '❌ FAIL'}")
            if result.issues:
                for issue in result.issues:
                    print(f"    ❌ {issue}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"    ⚠️ {warning}")

        except Exception as e:
            result.error_message = str(e)
            result.issues.append(f"Unexpected error: {e}")
            print(f"\n  ❌ Unexpected error: {e}")

        results.append(result)

        # Delay between models to avoid rate limiting
        time.sleep(2)

    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Token Counting Verification Test for Safety Confirmation V6"
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model to test (default: all models)',
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='List of models to test',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for the report (e.g., report.md)',
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help='OpenAI API key (default: from OPENAI_API_KEY env var)',
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1"),
        help=f'API base URL (default: https://aihubmix.com/v1)',
    )

    args = parser.parse_args()

    # Use local variables for API configuration
    api_key = args.api_key
    base_url = args.base_url

    if not api_key:
        print("Error: OPENAI_API_KEY not set. Please set the environment variable or use --api-key.")
        sys.exit(1)

    # Determine which models to test
    if args.model:
        models_to_test = [args.model]
    elif args.models:
        models_to_test = args.models
    else:
        models_to_test = DEFAULT_MODELS

    print(f"Token Counting Verification Test")
    print(f"=" * 60)
    print(f"API Endpoint: {base_url}")
    print(f"Models to test: {len(models_to_test)}")
    print(f"Output: {args.output or 'stdout'}")

    # Run tests
    results = run_tests(models_to_test, api_key=api_key, base_url=base_url, verbose=args.verbose, output=args.output)

    # Generate report
    report = generate_markdown_report(results, base_url=base_url, output_path=args.output)

    if not args.output:
        print("\n" + report)

    # Exit with error code if any tests failed
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
