import sys
import matplotlib.pyplot as plt # type: ignore
import re
from collections import defaultdict

import numpy as np

def parse_log(file_name: str):
    with open(file_name, "r") as f:
        log_lines = f.readlines()

    data = defaultdict(lambda: {"results":[], "generations": 0, "time": 0, "count": 0})
    
    pattern = re.compile(
        r"prc: hpc trc: (\d+) gen: (\d+) pop: (\d+) ftol: ([\deE\.-]+) time: ([\d\.]+) results: (\[.*\])"
    )
    
    for line in log_lines:
        match = pattern.match(line)
        if match:
            trc = int(match.group(1))
            gen = int(match.group(2))
            pop = int(match.group(3))
            ftol = float(match.group(4))
            time = float(match.group(5))
            results = eval(match.group(6))
            
            key = (trc, pop, ftol)
            if key not in data.keys() or len(data[key]["results"]) < len(results):
                data[key]["results"] = results
            data[key]["generations"] += gen
            data[key]["time"] += time
            data[key]["count"] += 1

    return data

def plot_results(key: tuple[str, str, str], data: dict, file_name: str):
    x, y = zip(*data["results"])
    count = data["count"]
    gen = round(data["generations"] / count, 2)
    time = round(data["time"] / count, 2)
    legend_text = f"mean generation: {gen}\nmean time: {time} s"

    plt.figure(figsize=(12, 9))
    plt.scatter(x, y, color="red")
    # for i in range(len(x)):
    #     plt.plot([x[i], x[i]], [min(y), y[i]], color='blue')
    # for i in range(len(y)):
    #     plt.plot([min(x), x[i]], [y[i], y[i]], color='green')
    plt.title(f"Optimization Results trc: {key[0]}, pop: {key[1]}, ftol: {key[2]}")
    plt.xlabel("Duration")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.legend([legend_text], loc="upper right", fontsize=10, frameon=True)
    plt.savefig(file_name + ".png")
    plt.close()

def plot_trc(data: dict, folder: str):
    considered_results = []
    for key, results in data.items():
        if key[1] == 50 and key[2] == 2.5e-05 and key[0] != 500:
            considered_results.append((key[0], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2)))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    trc, gen, time = zip(*sorted_considered_results)

    plt.figure(figsize=(12, 9))
    plt.yscale('log')
    plt.plot(trc, gen, color="blue", label="gen")
    plt.plot(trc, time, color="red", label="time [s]")
    plt.title(f"Analysis of pop: 50 ftol: 2.5e-05")
    plt.xlabel("traces")
    plt.xticks(trc, rotation=60)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(folder + "res_trc_gen_time.png")
    plt.close()

def plot_pop(data, folder: str, trc: int):
    considered_results = []
    for key, results in data.items():
        if key[0] == trc and key[2] == 2.5e-05:
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
    plt.tight_layout()
    plt.legend()
    plt.savefig(folder + f"res_pop_gen_time_{trc}.png")
    plt.close()

    plt.figure(figsize=(12, 9))
    for i, sol in enumerate(res):
        x, y = zip(*sol)
        plt.scatter(x, y, color=colors[i], label=f"{pop[i]}")
    plt.title(f"Analysis of trc: 400 ftol: 2.5e-05")
    plt.xlabel("Duration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(folder + f"res_pop_res_{trc}.png")
    plt.close()

def plot_ftol(data, folder: str):
    considered_results = []
    for key, results in data.items():
        if key[0] == 400 and key[1] == 50:
            considered_results.append((key[2], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2), results["results"]))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    colors = [(1, 0, 0, alpha) for alpha in np.linspace(0.3, 1, len(considered_results))]
    # colors = ['yellow', 'orange', 'red', 'green', 'blue', 'purple', 'black']
    ftol, gen, time, res = zip(*sorted_considered_results)
    tick_labels = [f"{x:.1e}" for x in ftol] 

    plt.figure(figsize=(12, 9))
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(ftol, gen, color="blue", label="gen")
    plt.plot(ftol, time, color="red", label="time [s]")
    plt.title(f"Analysis of trc: 400 pop: 50")
    plt.xlabel("ftol")
    plt.xticks(ftol, tick_labels, rotation=45, ha='right')  # Set ticks and labels
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.savefig(folder + "res_ftol_gen_time.png")
    plt.close()
    
    plt.figure(figsize=(12, 9))
    for i, sol in enumerate(res):
        x, y = zip(*sol)
        plt.scatter(x, y, color=colors[i], label=f"{ftol[i]}")
    plt.title(f"Analysis of trc: 400 pop: 50")
    plt.xlabel("Duration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(folder + "res_ftol_res.png")
    plt.close()


log_file = "ignored/sim_1.txt"
parsed_data = parse_log(log_file)

folder = "ignored/parsed_res/"

for key, data in parsed_data.items():
    plot_results(key, data, f"{folder}parsed_data_{key}")

plot_trc(parsed_data, folder)
plot_pop(parsed_data, folder, 400)
plot_pop(parsed_data, folder, 500)
plot_ftol(parsed_data, folder)