import json
import os
import sys

def load_json_from_env(env_var):
    data = os.environ.get(env_var)
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {env_var}")
        return None

def get_benchmark_map(data):
    if not data or 'benchmarks' not in data:
        return {}
    return {b['name']: b for b in data['benchmarks']}

def calculate_total_time(data):
    if not data or 'benchmarks' not in data:
        return float('inf')
    # Sum of real_time for all benchmarks
    return sum(b['real_time'] for b in data['benchmarks'])

def main():
    # Load current results
    try:
        with open('benchmark_results.json', 'r') as f:
            current_data = json.load(f)
    except FileNotFoundError:
        print("Error: benchmark_results.json not found")
        sys.exit(1)

    # Load stored results
    prev_data = load_json_from_env('PREV_BENCHMARK_JSON')
    best_data = load_json_from_env('BEST_BENCHMARK_JSON')
    best_sha = os.environ.get('BEST_BENCHMARK_SHA', 'N/A')
    current_sha = os.environ.get('GITHUB_SHA', 'unknown')

    current_map = get_benchmark_map(current_data)
    prev_map = get_benchmark_map(prev_data)
    best_map = get_benchmark_map(best_data)

    # Calculate scores (lower is better)
    current_score = calculate_total_time(current_data)
    best_score = calculate_total_time(best_data)

    is_new_best = current_score < best_score
    
    # Generate Summary Markdown
    summary = []
    summary.append("## 🚀 Benchmark Comparison")
    summary.append(f"**Current Commit:** `{current_sha[:7]}`")
    
    if is_new_best:
        summary.append(f"🎉 **New Best Record!** (Previous Best: `{best_sha[:7]}`)")
    else:
        summary.append(f"**Best Record:** `{best_sha[:7]}`")

    summary.append("\n| Benchmark | Current (ns) | vs Previous | vs Best |")
    summary.append("|---|---|---|---|")

    for name, curr_b in current_map.items():
        curr_time = curr_b['real_time']
        
        # Compare with Previous
        prev_str = "-"
        if name in prev_map:
            prev_time = prev_map[name]['real_time']
            diff = (curr_time - prev_time) / prev_time * 100
            icon = "🔴" if diff > 5 else ("🟢" if diff < -5 else "⚪")
            prev_str = f"{diff:+.2f}% {icon}"

        # Compare with Best
        best_str = "-"
        if name in best_map:
            best_time = best_map[name]['real_time']
            diff = (curr_time - best_time) / best_time * 100
            best_str = f"{diff:+.2f}%"

        summary.append(f"| `{name}` | {curr_time:.2f} | {prev_str} | {best_str} |")

    # Write summary to GITHUB_STEP_SUMMARY
    summary_text = "\n".join(summary)
    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(summary_text + "\n")
    
    print(summary_text)

    # Prepare data for saving
    # We minify the JSON to save space in variables
    def minify(data):
        if not data: return "{}"
        # Keep only essential fields to save space
        minified = {"benchmarks": []}
        for b in data.get("benchmarks", []):
            minified["benchmarks"].append({
                "name": b["name"],
                "real_time": b["real_time"]
            })
        return json.dumps(minified, separators=(',', ':'))

    # Save Previous (Current becomes Previous)
    with open('next_prev_benchmark.json', 'w') as f:
        f.write(minify(current_data))

    # Save Best
    if is_new_best:
        with open('next_best_benchmark.json', 'w') as f:
            f.write(minify(current_data))
        with open('next_best_sha.txt', 'w') as f:
            f.write(current_sha)
        print(f"::notice::New best benchmark achieved! Score: {current_score}")
    else:
        # Keep existing best (or create if none)
        if not best_data:
             with open('next_best_benchmark.json', 'w') as f:
                f.write(minify(current_data))
             with open('next_best_sha.txt', 'w') as f:
                f.write(current_sha)

if __name__ == "__main__":
    main()
