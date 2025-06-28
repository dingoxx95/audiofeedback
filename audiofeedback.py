#!/usr/bin/env python3
"""
Audio Feedback Analyzer - Entry Point
"""

import sys

try:
    from audiofeedback_core.app import main
except ImportError as e:
    print(f"Error importing audiofeedback_core: {e}")
    sys.exit(1)

if __name__ == "__main__":
    main()
