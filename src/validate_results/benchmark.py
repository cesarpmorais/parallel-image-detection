#!/usr/bin/env python3
"""
Benchmark script to run C++ inference over a dataset and gather timings.

Usage examples:
  python benchmark.py --bin ../../cpp/build/resnet18 --images ../../datasets --out results.csv --runs 5 --warmup 1

This script:
 - Preprocesses images using the same preprocessing as the reference model
 - Saves input to test_data/input.bin and input_shape.txt
 - Runs the specified C++ binary (single-threaded or GPU build)
 - Reads per-layer timings from src/validate_results/cpp_outputs/timings.csv
 - Reads final_output from src/validate_results/cpp_outputs/final_output.bin to extract top-1
 - Writes per-run data and aggregated statistics to a CSV
"""

import argparse
import os
import subprocess
import time
import glob
import csv
import numpy as np

from reference_model import preprocess_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DATA_DIR = os.path.join(ROOT, "test_data")
CPP_OUTPUTS = os.path.join(ROOT, "cpp_outputs")


def save_input(tensor, out_dir=TEST_DATA_DIR):
    os.makedirs(out_dir, exist_ok=True)
    data = tensor.detach().cpu().numpy().astype(np.float32)
    data.tofile(os.path.join(out_dir, "input.bin"))
    with open(os.path.join(out_dir, "input_shape.txt"), "w") as f:
        f.write(" ".join(map(str, data.shape)))


def read_timings(csv_path):
    if not os.path.exists(csv_path):
        return {}
    d = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if row[0] == "total":
                d["total"] = float(row[1])
            else:
                try:
                    d[row[0]] = float(row[1])
                except Exception:
                    d[row[0]] = 0.0
    return d


def read_output_top1(bin_path, shape_path=None):
    if not os.path.exists(bin_path):
        return None
    out = np.fromfile(bin_path, dtype=np.float32)
    if shape_path and os.path.exists(shape_path):
        with open(shape_path, "r") as f:
            shape = tuple(map(int, f.read().strip().split()))
        out = out.reshape(shape)
    # flatten
    if out.ndim > 1:
        out = out.flatten()
    idx = int(np.argmax(out))
    return idx


def run_binary(bin_path, cwd, timeout=None):
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [bin_path],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        wall = (time.perf_counter() - start) * 1000.0
        return (
            proc.returncode,
            wall,
            proc.stdout.decode("utf-8", errors="ignore"),
            proc.stderr.decode("utf-8", errors="ignore"),
        )
    except subprocess.TimeoutExpired:
        return -1, None, "", "timeout"


def collect_images(images_dir):
    # Walk directory recursively and match common image extensions (case-insensitive)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for root, _, filenames in os.walk(images_dir):
        for fn in filenames:
            _, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def aggregate_stats(values):
    arr = np.array(values)
    return (
        float(arr.mean()),
        float(arr.std()),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark C++ ResNet18 binaries over a dataset"
    )
    parser.add_argument(
        "--bin", type=str, required=True, help="Path to C++ executable to run"
    )
    parser.add_argument(
        "--images",
        type=str,
        default=os.path.join(ROOT, "..", "datasets"),
        help="Images directory",
    )
    parser.add_argument(
        "--out", type=str, default="benchmark_results.csv", help="Output CSV summary"
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per image")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs per image")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images to run (0 = all)",
    )
    # short alias for convenience
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        dest="max_images",
        help="Alias for --max-images (limit number of images to run)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose debug info (resolved binary path, commands)",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per run (s)")
    args = parser.parse_args()

    images = collect_images(args.images)
    if not images:
        print("No images found in", args.images)
        return

    # Resolve binary path: if user-supplied path doesn't exist, try common locations
    def resolve_binary(bin_path):
        if os.path.isabs(bin_path) and os.path.exists(bin_path):
            return bin_path
        # first try relative as given
        candidate = os.path.join(os.getcwd(), bin_path)
        if os.path.exists(bin_path):
            return bin_path
        if os.path.exists(candidate):
            return candidate

        # try common locations relative to repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        common = [
            os.path.join(repo_root, "cpp", "build", "resnet18"),
            os.path.join(repo_root, "cpp", "resnet18"),
            os.path.join(repo_root, "cpp", "resnet18.exe"),
            os.path.join(repo_root, "cpp", "build", "resnet18.exe"),
            os.path.join(repo_root, "cpp", "resnet18.bin"),
        ]
        for c in common:
            if os.path.exists(c):
                return c

        # fallback: return original (will fail later)
        return bin_path

    args.bin = resolve_binary(args.bin)

    cpp_cwd = os.path.abspath(os.path.join(ROOT, "..", "cpp"))
    timings_csv = os.path.join(CPP_OUTPUTS, "timings.csv")
    final_bin = os.path.join(CPP_OUTPUTS, "final_output.bin")
    final_shape = os.path.join(CPP_OUTPUTS, "final_output_shape.txt")

    rows = []
    # apply max-images limit if set
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    if args.verbose:
        print(f"Resolved binary path: {args.bin}")
    print(f"Found {len(images)} images. Running benchmark on binary: {args.bin}\n")

    for img_path in images:
        print("Image:", img_path)
        tensor = preprocess_image(img_path)
        save_input(tensor, TEST_DATA_DIR)

        # Warm-up
        for i in range(args.warmup):
            rc, wall, out, err = run_binary(args.bin, cwd=cpp_cwd, timeout=args.timeout)
            if rc != 0:
                print("Warmup run failed:", err)

        # Measured runs
        per_run_walls = []
        per_run_timings = []
        top1s = []
        for r in range(args.runs):
            rc, wall, out, err = run_binary(args.bin, cwd=cpp_cwd, timeout=args.timeout)
            if rc != 0:
                print(f"Run {r} failed for image {img_path}:", err)
                continue
            per_run_walls.append(wall)

            # read C++ timings
            t = read_timings(timings_csv)
            per_run_timings.append(t)

            top1 = read_output_top1(final_bin, final_shape)
            top1s.append(top1)

        if not per_run_walls:
            print("No successful runs for", img_path)
            continue

        # aggregate
        mean_wall, std_wall, median_wall, p95_wall, p99_wall = aggregate_stats(
            per_run_walls
        )

        # aggregate per-layer by averaging across runs
        layers = set()
        for d in per_run_timings:
            layers.update(d.keys())
        layer_means = {}
        for layer in layers:
            vals = [d.get(layer, 0.0) for d in per_run_timings]
            layer_means[layer] = float(np.mean(vals)) if vals else 0.0

        # top1 majority
        top1 = None
        if top1s:
            vals = [t for t in top1s if t is not None]
            if vals:
                top1 = int(np.bincount(vals).argmax())

        row = {
            "image": os.path.basename(img_path),
            "runs": len(per_run_walls),
            "mean_wall_ms": mean_wall,
            "std_wall_ms": std_wall,
            "median_wall_ms": median_wall,
            "p95_wall_ms": p95_wall,
            "p99_wall_ms": p99_wall,
            "top1": top1,
        }
        # add layer means
        for k, v in layer_means.items():
            row[f"layer_{k}"] = v

        rows.append(row)

    # write CSV
    keys = [
        "image",
        "runs",
        "mean_wall_ms",
        "std_wall_ms",
        "median_wall_ms",
        "p95_wall_ms",
        "p99_wall_ms",
        "top1",
    ]
    # include any layer keys found
    layer_keys = sorted([k for r in rows for k in r.keys() if k.startswith("layer_")])
    keys.extend(layer_keys)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    print("Benchmark complete. Results written to", args.out)


if __name__ == "__main__":
    main()
