import sys
import matplotlib.pyplot as plt # type: ignore
import re
from collections import defaultdict

def parse_log(lines):
    data = defaultdict(list)
    data = defaultdict(lambda: {"results":[], "generations": 0, "time": 0, "count": 0})
    
    pattern = re.compile(
        r"prc: hpc trc: (\d+) gen: (\d+) pop: (\d+) ftol: ([\deE\.-]+) time: ([\d\.]+) results: (\[.*\])"
    )
    
    for line in lines:
        match = pattern.match(line)
        if match:
            trc = int(match.group(1))
            gen = int(match.group(2))
            pop = int(match.group(3))
            ftol = float(match.group(4))
            time = float(match.group(5))
            results = eval(match.group(6))
            
            key = (trc, pop, ftol)
            data[key]["results"].extend(results)
            data[key]["generations"] += gen
            data[key]["time"] += time
            data[key]["count"] += 1

    return data

def plot_results(key, data, file_name="results_plot"):
    x, y = zip(*data["results"])
    count = data["count"]
    gen = round(data["generations"] / count, 2)
    time = round(data["time"] / count, 2)
    legend_text = f"Total simulations: {count}\nmean generation: {gen}\nmean time: {time} s"

    plt.figure(figsize=(12, 9))
    plt.scatter(x, y, color="red")
    plt.title(f"Optimization Results pop: {key[0]}, trc: {key[1]}, ftol: {key[2]}")
    plt.xlabel("Duration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.legend([legend_text], loc="upper right", fontsize=10, frameon=True)
    plt.savefig(file_name + ".png")
    plt.close()

log_file = sys.argv[1]
with open(log_file, "r") as f:
    log_lines = f.readlines()

parsed_data = parse_log(log_lines)
for key, data in parsed_data.items():
    plot_results(key, data, f"cluster/parsed_results/parsed_data_{key}")
