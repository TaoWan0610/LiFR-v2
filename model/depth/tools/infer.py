import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(ROOT.parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT.parents[1]))

from moge.model import load_model
from moge.utils.vis import colorize_depth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct Anytime MoGe2 inference from one RGB image and event voxel .npy files."
    )
    parser.add_argument("--pretrained", required=True, help="Path to MoGe2 checkpoint, e.g. /path/to/model.pt")
    parser.add_argument('--image', required=True, help='Input RGB image at t0.')
    parser.add_argument('--events', nargs='+', default=[], help='Chronological .npy voxels: context before t0, then one per warp.')
    parser.add_argument('--output', required=True, help='Output directory for depth.npy.')
    parser.add_argument('--visualize', action='store_true', help='Also save depth_vis.png.')
    parser.add_argument("--device", default="cuda:0", help="Torch device. Default: cuda:0.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 inference.")
    parser.add_argument("--fov-x", type=float, default=None, help="Optional horizontal FoV in degrees.")
    parser.add_argument("--num-tokens", type=int, default=None, help="Base ViT token count. Default uses model resolution_level.")
    parser.add_argument("--resolution-level", type=int, default=9, help="Resolution level [0-9]. Default: 9.")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize for image and event voxels, e.g. --resize 640 440.",
    )
    parser.add_argument("--apply-mask", action="store_true", help="Apply predicted mask to depth output.")
    return parser


def read_image(path: Path, resize: Optional[Tuple[int, int]]) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if resize is not None:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    return image


def read_event(path: Path, resize: Optional[Tuple[int, int]], expected_hw: Tuple[int, int]) -> np.ndarray:
    event = np.load(path, allow_pickle=False).astype(np.float32)
    if event.ndim != 3:
        raise ValueError(f"Expected event voxel to be 3D, got {event.shape} from {path}")

    if event.shape[0] in (4, 5, 10, 20):
        event_chw = event
    elif event.shape[-1] in (4, 5, 10, 20):
        event_chw = event.transpose(2, 0, 1)
    else:
        raise ValueError(f"Cannot infer event channel axis from shape {event.shape} in {path}")

    event_chw = np.nan_to_num(event_chw, nan=0.0, posinf=0.0, neginf=0.0)

    if resize is not None:
        width, height = resize
        event_chw = np.stack(
            [cv2.resize(ch, (width, height), interpolation=cv2.INTER_LINEAR) for ch in event_chw],
            axis=0,
        ).astype(np.float32)

    h, w = event_chw.shape[-2:]
    if (h, w) != expected_hw:
        raise ValueError(f"Event size {h}x{w} does not match image size {expected_hw[0]}x{expected_hw[1]} for {path}")
    return event_chw


def tensor_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    resize = tuple(args.resize) if args.resize is not None else None
    model = load_model(args.pretrained).to(device).eval()
    expected = model.num_warps + 1 if model.anytime_enabled else 0
    if len(args.events) != expected:
        raise ValueError(f'This checkpoint requires {expected} event voxels; got {len(args.events)}')
    image = read_image(Path(args.image), resize)
    image_t = tensor_image(image, device)
    events = [torch.from_numpy(read_event(Path(path), resize, image.shape[:2])).unsqueeze(0).to(device)
              for path in args.events]
    kwargs = {'event_sequence': events} if events else {}

    with torch.inference_mode():
        output = model.infer(
            image_t,
            fov_x=args.fov_x,
            apply_mask=args.apply_mask,
            num_tokens=args.num_tokens,
            resolution_level=args.resolution_level,
            use_fp16=args.fp16,
            **kwargs,
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth = output['depth'].detach().float().cpu().numpy()
    np.save(out_dir / 'depth.npy', depth)
    if args.visualize:
        cv2.imwrite(str(out_dir / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
    print(f'Saved depth to {out_dir / "depth.npy"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
