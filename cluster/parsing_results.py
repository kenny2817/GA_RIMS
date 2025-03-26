import sys
import matplotlib.pyplot as plt # type: ignore
import re
from collections import defaultdict

import numpy as np

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
    plt.title(f"Optimization Results trc: {key[0]}, pop: {key[1]}, ftol: {key[2]}")
    plt.xlabel("Duration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.legend([legend_text], loc="upper right", fontsize=10, frameon=True)
    plt.savefig(file_name + ".png")
    plt.close()

def plot_trc(data, file_name="cluster/parsed_results/res_trc"):
    considered_results = []
    for key, results in data.items():
        if key[1] == 50 and key[2] == 2.5e-05:
            considered_results.append((key[0], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2)))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    trc, gen, time = zip(*sorted_considered_results)

    plt.figure(figsize=(12, 9))
    plt.yscale('log')
    plt.plot(trc, gen, color="blue", label="gen")
    plt.plot(trc, time, color="red", label="time [s]")
    plt.title(f"Analysis of pop: 50 ftol: 2.5e-05")
    plt.xlabel("traces")
    plt.xticks(trc)
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name + "_gen_time.png")
    plt.close()

def plot_pop(data, file_name="cluster/parsed_results/res_pop"):
    considered_results = []
    for key, results in data.items():
        if key[0] == 400 and key[2] == 2.5e-05:
            considered_results.append((key[1], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2), results["results"]))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    colors = [(1, 0, 0, alpha) for alpha in np.linspace(0.3, 1, len(considered_results))]
    pop, gen, time, res = zip(*sorted_considered_results)

    plt.figure(figsize=(12, 9))
    plt.yscale('log')
    plt.plot(pop, gen, color="blue", label="gen")
    plt.plot(pop, time, color="red", label="time [s]")
    plt.title(f"Analysis of trc: 400 ftol: 2.5e-05")
    plt.xlabel("pop size")
    plt.xticks(pop)
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name + "_gen_time.png")
    plt.close()

    plt.figure(figsize=(12, 9))
    for i, sol in enumerate(res):
        x, y = zip(*sol)
        plt.scatter(x, y, color=colors[i], label=f"{pop[i]}")
    plt.title(f"Analysis of trc: 400 ftol: 2.5e-05")
    plt.xlabel("Duration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name + "_res.png")
    plt.close()

def plot_ftol(data, file_name="cluster/parsed_results/res_ftol"):
    considered_results = []
    for key, results in data.items():
        if key[0] == 400 and key[1] == 50:
            considered_results.append((key[2], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2), results["results"]))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    colors = [(1, 0, 0, alpha) for alpha in np.linspace(0.3, 1, len(considered_results))]
    ftol, gen, time, res = zip(*sorted_considered_results)

    plt.figure(figsize=(12, 9))
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(ftol, gen, color="blue", label="gen")
    plt.plot(ftol, time, color="red", label="time [s]")
    plt.title(f"Analysis of trc: 400 pop: 50")
    plt.xlabel("ftol")
    plt.xticks(ftol)
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name + "_gen_time.png")
    plt.close()
    
    plt.figure(figsize=(12, 9))
    for i, sol in enumerate(res):
        x, y = zip(*sol)
        plt.scatter(x, y, color=colors[i], label=f"{ftol[i]}")
    plt.title(f"Analysis of trc: 400 pop: 50")
    plt.xlabel("Duration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name + "_res.png")
    plt.close()


log_file = sys.argv[1]
with open(log_file, "r") as f:
    log_lines = f.readlines()

parsed_data = parse_log(log_lines)
for key, data in parsed_data.items():
    plot_results(key, data, f"cluster/parsed_results/parsed_data_{key}")

plot_trc(parsed_data)
plot_pop(parsed_data)
plot_ftol(parsed_data)