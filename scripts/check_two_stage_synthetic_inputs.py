#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.synthetic_two_stage_utils import validate_synthetic_scene  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--lr_image_dir", required=True)
    parser.add_argument("--hr_image_dir", required=True)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()
    result = validate_synthetic_scene(
        args.scene_dir,
        args.lr_image_dir,
        args.hr_image_dir,
        args.scale,
        True,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
