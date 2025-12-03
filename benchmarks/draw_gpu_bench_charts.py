import json
import re
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
from pathlib import Path


name_re = re.compile(r'(?P<bench>[^/]+)/(?P<size>\d+)/manual_time')

def load_bench_file(path, label):
    """
    读取一个 Google Benchmark JSON 文件，解析出
    (benchmark_name, size) -> {label: {bytes_per_second, real_time}}
    """
    with open(path, "r") as f:
        data = json.load(f)

    results = defaultdict(dict)  # key: (bench, size) -> label -> metrics
    for entry in data.get("benchmarks", []):
        name = entry.get("name", "")
        m = name_re.match(name)
        if not m:
            continue

        bench = m.group("bench")       # e.g. BM_Pack5_CUDA
        size = int(m.group("size"))    # e.g. 65536

        metrics = {
            "bytes_per_second": entry.get("bytes_per_second", None),
            "real_time": entry.get("real_time", None),  # already in ms
        }
        results[(bench, size)][label] = metrics

    return results


def merge_results(result_list):
    """
    把多个文件的结果合并：
    (bench, size) -> label -> metrics
    再拆成：
    bench -> size -> label -> metrics
    """
    merged = defaultdict(dict)
    for res in result_list:
        for key, val in res.items():
            if key not in merged:
                merged[key] = {}
            merged[key].update(val)

    by_bench = defaultdict(dict)  # bench -> size -> label -> metrics
    for (bench, size), label_dict in merged.items():
        by_bench[bench][size] = label_dict
    return by_bench


def plot_throughput(by_bench, labels, out_dir: Path):
    """
    对每个 bench，画 bytes_per_second(GB/s) vs size 的对比曲线
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench, sizes_dict in by_bench.items():
        # Only consider cases with data
        sizes = sorted(sizes_dict.keys())
        if not sizes:
            continue

        plt.figure()
        for label in labels:
            y = []
            x = []
            for s in sizes:
                metrics_for_size = sizes_dict[s].get(label)
                if not metrics_for_size:
                    # 这个 label 在这个 size 没有数据
                    continue
                bps = metrics_for_size.get("bytes_per_second")
                if bps is None:
                    continue
                x.append(s)
                # 转成 GB/s
                y.append(bps / 1e9)

            if x:
                plt.plot(x, y, marker="o", label=label)

        if not plt.gca().has_data():
            plt.close()
            continue

        plt.xscale("log", base=2)
        plt.xlabel("Problem size")
        plt.ylabel("Throughput (GB/s)")
        plt.title(f"{bench} - bytes_per_second")
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"{bench}_throughput.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def plot_latency(by_bench, labels, out_dir: Path):
    """
    对每个 bench，画 real_time(ms) vs size 的对比曲线
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench, sizes_dict in by_bench.items():
        sizes = sorted(sizes_dict.keys())
        if not sizes:
            continue

        plt.figure()
        for label in labels:
            y = []
            x = []
            for s in sizes:
                metrics_for_size = sizes_dict[s].get(label)
                if not metrics_for_size:
                    continue
                rt = metrics_for_size.get("real_time")
                if rt is None:
                    continue
                x.append(s)
                y.append(rt)  # 已经是 ms

            if x:
                plt.plot(x, y, marker="o", label=label)

        if not plt.gca().has_data():
            plt.close()
            continue

        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Problem size")
        plt.ylabel("Latency (ms, real_time)")
        plt.title(f"{bench} - real_time")
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"{bench}_latency.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance of two Google Benchmark JSON results with matplotlib."
    )
    parser.add_argument("json_files", nargs="+", help="Paths to benchmark JSON files (2 or more).")
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Labels for each JSON file in the same order. If not set, file names will be used.",
    )
    parser.add_argument(
        "--out-dir",
        default="plots",
        help="Output directory to save PNG plots (default: plots)",
    )
    args = parser.parse_args()

    if len(args.json_files) < 2:
        raise SystemExit("Please provide at least two JSON files to compare.")

    if args.labels is not None and len(args.labels) != len(args.json_files):
        raise SystemExit("Number of labels must match number of JSON files.")

    out_dir = Path(args.out_dir)

    labels = args.labels
    if labels is None:
        labels = [Path(p).stem for p in args.json_files]

    all_results = []
    for path, label in zip(args.json_files, labels):
        res = load_bench_file(path, label)
        all_results.append(res)

    by_bench = merge_results(all_results)

    print("Benchmarks found:")
    for bench in sorted(by_bench.keys()):
        print("  ", bench)

    plot_throughput(by_bench, labels, out_dir / "throughput")
    plot_latency(by_bench, labels, out_dir / "latency")


if __name__ == "__main__":
    main()
