#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.x8_scale_utils import validate_colmap_scene_x8  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--lr_image_dir", required=True)
    parser.add_argument("--hr_image_dir", required=True)
    parser.add_argument("--scale", type=int, choices=(8,), default=8)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    result = validate_colmap_scene_x8(
        args.scene_dir,
        args.lr_image_dir,
        args.hr_image_dir,
        args.scale,
        args.eval,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
