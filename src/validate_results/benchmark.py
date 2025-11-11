#!/usr/bin/env python3
"""
Benchmark script to preprocess images and run C++ inference over them in one batch.

Usage examples:
  python benchmark.py --bin ../../cpp/build/resnet18 --images ../../datasets --out results.csv --max-images 100
  python benchmark.py --bin ../../cpp/build/resnet18 --images ../../datasets --out results.csv -n 10 --repeat 3

This script:
 - Preprocesses images into .bin and .txt shape files in a temp directory
 - Calls C++ binary once with --images-dir to process all images in batch (no repeated startup)
 - Collects per-image timing CSVs and final outputs from C++ binary
 - Writes aggregated summary CSV with per-image stats and per-layer timings
"""

import argparse
import os
import subprocess
import time
import glob
import csv
import numpy as np
import tempfile
import shutil
import torch

from reference_model import preprocess_image, ResNet18Reference

# ROOT should point to the directory containing benchmark.py (validate_results/)
ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(ROOT, "test_data")
CPP_OUTPUTS = os.path.join(ROOT, "cpp_outputs")


def save_input(tensor, out_path, out_dir=None):
    """Save tensor as .bin and .txt shape file."""
    if out_dir is None:
        out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    data = tensor.detach().cpu().numpy().astype(np.float32)
    data.tofile(out_path)
    shape_path = out_path.rsplit(".", 1)[0] + "_shape.txt"
    with open(shape_path, "w") as f:
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


def run_binary(bin_path, cwd, images_dir=None, max_images=0, timeout=None):
    """Run C++ binary with optional --images-dir argument."""
    start = time.perf_counter()
    try:
        cmd = [bin_path]
        if images_dir:
            cmd.extend(["--images-dir", images_dir, "--max-images", str(max_images)])
        proc = subprocess.run(
            cmd,
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


def validate_predictions(image_path, cpp_output_bin, cpp_output_shape):
    """
    Compare C++ predictions against reference PyTorch model.
    Returns True if top-1 predictions match, False otherwise.
    """
    try:
        # Get reference prediction
        ref_model = ResNet18Reference()
        with torch.no_grad():
            input_tensor = preprocess_image(image_path)
            ref_output = ref_model.model(input_tensor)
        ref_top1 = int(torch.argmax(ref_output[0]).item())

        # Get C++ prediction
        cpp_output = np.fromfile(cpp_output_bin, dtype=np.float32)
        if os.path.exists(cpp_output_shape):
            with open(cpp_output_shape, "r") as f:
                shape = tuple(map(int, f.read().strip().split()))
            cpp_output = cpp_output.reshape(shape)
        if cpp_output.ndim > 1:
            cpp_output = cpp_output.flatten()
        cpp_top1 = int(np.argmax(cpp_output))

        # Compare
        match = ref_top1 == cpp_top1
        return match, ref_top1, cpp_top1
    except Exception as e:
        print(f"  Validation error: {e}")
        return False, None, None


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
        description="Benchmark C++ ResNet18 over dataset images"
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
    parser.add_argument(
        "-n",
        "--max-images",
        type=int,
        default=0,
        dest="max_images",
        help="Limit number of images (0 = all)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat inference for each image (to amortize startup cost)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose debug info (resolved binary path, commands)",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per run (s)")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare C++ predictions against reference PyTorch model",
    )
    args = parser.parse_args()

    # collect image files
    all_images = collect_images(args.images)
    if not all_images:
        print("No images found in", args.images)
        return

    if args.max_images > 0:
        all_images = all_images[: args.max_images]

    print(f"Found {len(all_images)} images. Creating temp preprocessed inputs...")

    # create temp dir for preprocessed inputs
    temp_inputs = tempfile.mkdtemp(prefix="resnet18_inputs_")
    try:
        # preprocess all images into .bin files in temp dir
        for i, img_path in enumerate(all_images):
            fname = os.path.basename(img_path)
            stem = os.path.splitext(fname)[0]
            out_bin = os.path.join(temp_inputs, f"{stem}.bin")
            try:
                tensor = preprocess_image(img_path)
                save_input(tensor, out_bin)
                if args.verbose:
                    print(f"  [{i+1}/{len(all_images)}] Preprocessed {fname}")
            except Exception as e:
                print(f"Warning: failed to preprocess {fname}: {e}")

        # resolve binary path
        def resolve_binary(bin_path):
            if os.path.isabs(bin_path) and os.path.exists(bin_path):
                return bin_path
            if os.path.exists(bin_path):
                return bin_path
            candidate = os.path.join(os.getcwd(), bin_path)
            if os.path.exists(candidate):
                return candidate
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            common = [
                os.path.join(repo_root, "cpp", "build", "resnet18"),
                os.path.join(repo_root, "cpp", "resnet18"),
                os.path.join(repo_root, "cpp", "build", "resnet18.exe"),
                os.path.join(repo_root, "cpp", "resnet18.exe"),
            ]
            for c in common:
                if os.path.exists(c):
                    return c
            return bin_path

        binary = resolve_binary(args.bin)
        # ensure binary path is absolute for subprocess
        if not os.path.isabs(binary):
            binary = os.path.abspath(binary)
        if args.verbose:
            print(f"Resolved binary: {binary}")
        print(
            f"Running C++ binary on {len(all_images)} preprocessed images (--repeat {args.repeat})..."
        )

        # cpp_cwd should be the cpp build directory (parent of validate_results)
        cpp_cwd = os.path.abspath(os.path.join(ROOT, "..", "..", "cpp"))
        rc, wall, out, err = run_binary(
            binary,
            cwd=cpp_cwd,
            images_dir=temp_inputs,
            max_images=len(all_images),
            timeout=args.timeout,
        )

        if rc != 0:
            print(f"C++ binary failed: {err}")
            return

        print(f"C++ binary completed (wall time: {wall:.1f} ms)")
        if args.verbose:
            print("Binary stdout:", out)

        # collect per-image timings and outputs
        rows = []
        validation_results = []
        for i, img_path in enumerate(all_images):
            fname = os.path.basename(img_path)
            stem = os.path.splitext(fname)[0]

            # read per-image timings
            timings_csv = os.path.join(CPP_OUTPUTS, f"timings_{stem}.csv")
            timings = read_timings(timings_csv)

            # read final output and extract top-1
            final_bin = os.path.join(CPP_OUTPUTS, f"final_output_{stem}.bin")
            final_shape = os.path.join(CPP_OUTPUTS, f"final_output_{stem}_shape.txt")
            top1 = read_output_top1(final_bin, final_shape)

            # validate if requested
            valid = None
            if args.validate:
                valid, ref_top1, cpp_top1 = validate_predictions(
                    img_path, final_bin, final_shape
                )
                status = "✓ PASS" if valid else "✗ FAIL"
                validation_results.append(
                    {
                        "image": fname,
                        "status": status,
                        "reference_top1": ref_top1,
                        "cpp_top1": cpp_top1,
                    }
                )

            row = {"image": fname, "top1": top1}
            if args.validate:
                row["valid"] = valid
            # add layer timings
            for layer, t_ms in timings.items():
                row[f"layer_{layer}"] = t_ms

            rows.append(row)

        # write results
        keys = ["image", "top1"]
        if args.validate:
            keys.append("valid")
        layer_keys = sorted(
            set(k for r in rows for k in r.keys() if k.startswith("layer_"))
        )
        keys.extend(layer_keys)

        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in keys})

        print(f"Results written to {args.out}")

        # Print validation summary if requested
        if args.validate and validation_results:
            print("\n=== Validation Results ===")
            passed = sum(1 for v in validation_results if v["status"].startswith("✓"))
            total = len(validation_results)
            print(f"Passed: {passed}/{total}")
            for v in validation_results:
                print(
                    f"  {v['image']:<40} {v['status']:<10} (ref={v['reference_top1']}, cpp={v['cpp_top1']})"
                )

    finally:
        # cleanup temp dir
        if os.path.exists(temp_inputs):
            shutil.rmtree(temp_inputs)
            if args.verbose:
                print(f"Cleaned up temp inputs: {temp_inputs}")


if __name__ == "__main__":
    main()
