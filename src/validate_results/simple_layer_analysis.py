#!/usr/bin/env python3
"""
Simple Layer Analysis for ResNet18 Performance
Works with existing timing infrastructure - minimal changes needed
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use("default")
sns.set_palette("tab10")

ROOT = os.path.abspath("../../")
OUTPUTS_DIR = os.path.join(ROOT, "src/validate_results/cpp_outputs")


def load_timing_data():
    """Load timing data from all implementations"""
    timing_files = {
        "CPU": os.path.join(OUTPUTS_DIR, "cpu", "timings.csv"),
        "OpenMP": os.path.join(OUTPUTS_DIR, "openmp", "timings.csv"),
        "CUDA": os.path.join(OUTPUTS_DIR, "cuda", "timings.csv"),
    }

    data = {}
    for impl, file_path in timing_files.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Convert to expected format - your current format is layer,time_ms
                # We'll simulate image_id=1 and convert ms to microseconds
                if "layer" in df.columns and "time_ms" in df.columns:
                    # Check for negative timings and convert them to positive
                    negative_timings = df[df["time_ms"] < 0]
                    negative_count = len(negative_timings)

                    if negative_count > 0:
                        print(
                            f"[INFO] Converting {negative_count} negative timing(s) to positive in {impl}:"
                        )
                        for _, row in negative_timings.iterrows():
                            print(
                                f"  {row['layer']}: {row['time_ms']:.3f} ms -> {abs(row['time_ms']):.3f} ms"
                            )

                    # Filter out 'total' row and any empty/invalid rows
                    df_clean = df[
                        (df["layer"] != "total")
                        & (df["layer"].notna())
                        & (df["layer"] != "")
                    ].copy()

                    # Convert negative timings to positive
                    df_clean["time_ms"] = df_clean["time_ms"].abs()

                    df_converted = pd.DataFrame(
                        {
                            "layer_name": df_clean["layer"],
                            "image_id": 1,
                            "duration_us": df_clean["time_ms"]
                            * 1000,  # Convert ms to microseconds
                        }
                    )
                    data[impl] = df_converted
                    print(f"[OK] Loaded {len(df_converted)} timing records for {impl}")
                else:
                    print(f"[WARN] Unexpected format in {file_path}")
            except Exception as e:
                print(f"[WARN] Failed to load {impl} data: {e}")
        else:
            print(f"[WARN] Timing file not found: {file_path}")

    return data


def analyze_layer_performance(data):
    """Analyze performance by layer"""
    if not data:
        print("[ERROR] No timing data available")
        return None

    # Calculate mean timing per layer for each implementation
    summary = []
    for impl_name, df in data.items():
        layer_stats = (
            df.groupby("layer_name")["duration_us"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        layer_stats["implementation"] = impl_name
        summary.append(layer_stats)

    summary_df = pd.concat(summary, ignore_index=True)
    return summary_df


def plot_layer_comparison(summary_df):
    """Create layer performance comparison with both linear and log scales"""
    if summary_df is None:
        return

    # Pivot for easier plotting
    pivot_df = summary_df.pivot(
        index="layer_name", columns="implementation", values="mean"
    )

    # Reorder columns to CPU/OpenMP/CUDA
    desired_order = ["CPU", "OpenMP", "CUDA"]
    available_cols = [col for col in desired_order if col in pivot_df.columns]
    pivot_df = pivot_df[available_cols]

    # Create two separate figures: linear and log scale

    # === FIGURE 1: LINEAR SCALE ===
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle(
        "Layer Performance Analysis - Linear Scale", fontsize=16, fontweight="bold"
    )

    # 1. Bar chart comparison (linear scale)
    ax1 = axes1[0, 0]
    pivot_df.plot(kind="bar", ax=ax1, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax1.set_title("Per-Layer Execution Time (Linear Scale)", fontweight="bold")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean Execution Time (μs)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(title="Implementation")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Speedup analysis
    ax2 = axes1[0, 1]
    if "CPU" in pivot_df.columns:
        speedup_data = {}
        for impl in ["OpenMP", "CUDA"]:
            if impl in pivot_df.columns:
                # Calculate speedup safely (avoid division by zero)
                speedup = pivot_df["CPU"] / pivot_df[impl].replace(0, 1e-6)
                speedup_data[f"{impl} vs CPU"] = speedup

        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            speedup_df.plot(kind="bar", ax=ax2, color=["#ff7f0e", "#2ca02c"])
            ax2.set_title("Speedup vs CPU by Layer", fontweight="bold")
            ax2.set_xlabel("Layer")
            ax2.set_ylabel("Speedup Factor")
            ax2.tick_params(axis="x", rotation=45)
            ax2.legend()
            ax2.grid(axis="y", alpha=0.3)

    # 3. Performance breakdown (CUDA)
    ax3 = axes1[1, 0]
    if "CUDA" in summary_df["implementation"].values:
        cuda_data = summary_df[summary_df["implementation"] == "CUDA"]
        cuda_data_filtered = cuda_data[cuda_data["mean"] > 0]
        if not cuda_data_filtered.empty:
            # Sort by timing for better visualization
            cuda_data_sorted = cuda_data_filtered.sort_values("mean", ascending=False)

            # Create custom colors for better distinction
            colors = plt.cm.Set3(np.linspace(0, 1, len(cuda_data_sorted)))

            wedges, texts, autotexts = ax3.pie(
                cuda_data_sorted["mean"],
                labels=None,  # Remove labels from pie to avoid overlap
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                pctdistance=0.85,
                textprops={"fontsize": 9, "fontweight": "bold"},
            )

            # Add legend with layer names
            ax3.legend(
                wedges,
                cuda_data_sorted["layer_name"],
                title="Layers",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=10,
            )

            ax3.set_title(
                "CUDA - Time Distribution by Layer", fontweight="bold", pad=20
            )

    # 4. Performance summary table
    ax4 = axes1[1, 1]
    ax4.axis("off")
    table_data = []
    for impl in pivot_df.columns:
        total_time = pivot_df[impl].sum()
        table_data.append([impl, f"{total_time:.0f} μs"])

    if "CPU" in pivot_df.columns:
        for impl in pivot_df.columns:
            if impl != "CPU":
                total_speedup = pivot_df["CPU"].sum() / pivot_df[impl].sum()
                table_data.append([f"{impl} Speedup", f"{total_speedup:.1f}x"])

    table_text = "Performance Summary:\n\n"
    for row in table_data:
        table_text += f"{row[0]}: {row[1]}\n"

    ax4.text(
        0.1,
        0.9,
        table_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plot_file1 = os.path.join(os.path.dirname(__file__), "layer_performance_linear.png")
    plt.savefig(plot_file1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Linear scale plot saved to {plot_file1}")

    # === FIGURE 2: LOG SCALE ===
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(
        "Layer Performance Analysis - Logarithmic Scale", fontsize=16, fontweight="bold"
    )

    # 1. Bar chart comparison (log scale)
    ax1 = axes2[0, 0]
    # Filter out zero values for log scale
    pivot_df_log = pivot_df.replace(0, np.nan)
    pivot_df_log.plot(
        kind="bar", ax=ax1, logy=True, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    ax1.set_title("Per-Layer Execution Time (Log Scale)", fontweight="bold")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean Execution Time (μs) - Log Scale")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(title="Implementation")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Detailed speedup with numbers
    ax2 = axes2[0, 1]
    if "CPU" in pivot_df.columns and "CUDA" in pivot_df.columns:
        cuda_speedup = pivot_df["CPU"] / pivot_df["CUDA"].replace(0, 1e-6)
        bars = ax2.bar(
            cuda_speedup.index, cuda_speedup.values, color="#2ca02c", alpha=0.7
        )
        ax2.set_title("CUDA Speedup vs CPU (with values)", fontweight="bold")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Speedup Factor")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(axis="y", alpha=0.3)

        # Add speedup values as text on bars
        for bar, speedup in zip(bars, cuda_speedup.values):
            if speedup > 0:  # Only show valid speedups
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 2,
                    f"{speedup:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    # 3. Implementation comparison (stacked)
    ax3 = axes2[1, 0]
    pivot_df_log.T.plot(kind="bar", ax=ax3, logy=True, width=0.8)
    ax3.set_title("Execution Time by Implementation (Log Scale)", fontweight="bold")
    ax3.set_xlabel("Implementation")
    ax3.set_ylabel("Execution Time (μs) - Log Scale")
    ax3.tick_params(axis="x", rotation=0)
    ax3.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # 4. Efficiency analysis
    ax4 = axes2[1, 1]
    if "CPU" in pivot_df.columns:
        efficiency_data = []
        layer_names = []

        for layer in pivot_df.index:
            cpu_time = pivot_df.loc[layer, "CPU"]
            if cpu_time > 0:
                layer_names.append(layer)
                row_data = [cpu_time]

                for impl in ["OpenMP", "CUDA"]:
                    if impl in pivot_df.columns:
                        impl_time = pivot_df.loc[layer, impl]
                        efficiency = (cpu_time / impl_time) if impl_time > 0 else 0
                        row_data.append(efficiency)
                    else:
                        row_data.append(0)

                efficiency_data.append(row_data[1:])  # Skip CPU baseline

        if efficiency_data:
            efficiency_df = pd.DataFrame(
                efficiency_data,
                index=layer_names,
                columns=[col for col in ["OpenMP", "CUDA"] if col in pivot_df.columns],
            )
            efficiency_df.plot(kind="bar", ax=ax4, color=["#ff7f0e", "#2ca02c"])
            ax4.set_title("Parallel Efficiency by Layer", fontweight="bold")
            ax4.set_xlabel("Layer")
            ax4.set_ylabel("Speedup Factor")
            ax4.tick_params(axis="x", rotation=45)
            ax4.legend(title="vs CPU")
            ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_file2 = os.path.join(os.path.dirname(__file__), "layer_performance_log.png")
    plt.savefig(plot_file2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Log scale plot saved to {plot_file2}")

    # Also create the original combined plot for backward compatibility
    plot_file_orig = os.path.join(
        os.path.dirname(__file__), "layer_performance_analysis.png"
    )

    # Quick combined version
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Use the log version for main comparison
    pivot_df_log.plot(
        kind="bar", ax=axes[0, 0], logy=True, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    axes[0, 0].set_title("Per-Layer Execution Time (Log Scale)", fontweight="bold")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Mean Execution Time (μs)")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].legend(title="Implementation")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Speedup chart
    if "CPU" in pivot_df.columns and speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        speedup_df.plot(kind="bar", ax=axes[0, 1], color=["#ff7f0e", "#2ca02c"])
        axes[0, 1].set_title("Speedup vs CPU by Layer", fontweight="bold")
        axes[0, 1].set_xlabel("Layer")
        axes[0, 1].set_ylabel("Speedup Factor")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(axis="y", alpha=0.3)

    # Pie chart
    if "CUDA" in summary_df["implementation"].values:
        cuda_data = summary_df[summary_df["implementation"] == "CUDA"]
        cuda_data_filtered = cuda_data[cuda_data["mean"] > 0]
        if not cuda_data_filtered.empty:
            cuda_data_sorted = cuda_data_filtered.sort_values("mean", ascending=False)
            colors = plt.cm.Set3(np.linspace(0, 1, len(cuda_data_sorted)))

            wedges, texts, autotexts = axes[1, 0].pie(
                cuda_data_sorted["mean"],
                labels=None,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                pctdistance=0.85,
                textprops={"fontsize": 9, "fontweight": "bold"},
            )
            axes[1, 0].legend(
                wedges,
                cuda_data_sorted["layer_name"],
                title="Layers",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=8,
            )
            axes[1, 0].set_title("CUDA - Time Distribution by Layer", fontweight="bold")

    # Summary table
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.1,
        0.9,
        table_text,
        transform=axes[1, 1].transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.savefig(plot_file_orig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Combined analysis plot saved to {plot_file_orig}")


def print_performance_insights(summary_df, data):
    """Print key performance insights"""
    if summary_df is None:
        return

    print("\n" + "=" * 60)
    print("🚀 PERFORMANCE ANALYSIS INSIGHTS")
    print("=" * 60)

    # Overall speedup
    total_times = {}
    for impl_name, df in data.items():
        total_time = df["duration_us"].sum() / 1000.0  # Convert to ms
        total_times[impl_name] = total_time

    if "CPU" in total_times:
        print(f"\n📊 Overall Performance:")
        for impl, time_ms in total_times.items():
            if impl != "CPU":
                speedup = total_times["CPU"] / time_ms
                print(f"  {impl}: {time_ms:.1f}ms ({speedup:.1f}x speedup)")
            else:
                print(f"  {impl}: {time_ms:.1f}ms (baseline)")

    # Layer-wise insights
    print(f"\n🎯 Layer-wise Analysis:")

    # Find best GPU layers
    if (
        "CUDA" in summary_df["implementation"].values
        and "CPU" in summary_df["implementation"].values
    ):
        cpu_data = summary_df[summary_df["implementation"] == "CPU"].set_index(
            "layer_name"
        )
        cuda_data = summary_df[summary_df["implementation"] == "CUDA"].set_index(
            "layer_name"
        )

        speedups = []
        for layer in cpu_data.index:
            if layer in cuda_data.index:
                speedup = cpu_data.loc[layer, "mean"] / cuda_data.loc[layer, "mean"]
                speedups.append((layer, speedup))

        speedups.sort(key=lambda x: x[1], reverse=True)

        print(f"  🏆 Top 3 GPU Accelerated Layers:")
        for layer, speedup in speedups[:3]:
            print(f"    {layer}: {speedup:.1f}x speedup")

        print(f"  ⚠️  Bottom 3 GPU Layers:")
        for layer, speedup in speedups[-3:]:
            print(f"    {layer}: {speedup:.1f}x speedup")

    print(f"\n💡 Optimization Recommendations:")
    print(f"  - Focus on convolution layers for maximum impact")
    print(f"  - Small layers may benefit from kernel fusion")
    print(f"  - Memory-bound operations need bandwidth optimization")


def main():
    """Main analysis function"""
    print("🔍 ResNet18 Layer Performance Analysis")
    print("=" * 50)

    # Load timing data
    data = load_timing_data()

    if not data:
        print("\n❌ No timing data found!")
        print("Make sure to run your benchmark first:")
        print("  cd src/validate_results")
        print("  python benchmark_parallel_vs_sequential.py 10")
        return

    # Analyze performance
    summary_df = analyze_layer_performance(data)

    if summary_df is not None:
        # Create visualizations
        plot_layer_comparison(summary_df)

        # Print insights
        print_performance_insights(summary_df, data)

        # Save summary
        summary_file = os.path.join(
            os.path.dirname(__file__), "layer_performance_summary.csv"
        )
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[OK] Summary data saved to {summary_file}")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
