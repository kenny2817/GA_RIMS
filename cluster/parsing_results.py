import matplotlib.pyplot as plt # type: ignore
from scipy.optimize import curve_fit # type: ignore
import re
from collections import defaultdict

import numpy as np

def parse_log_0(file_name: str, raw_pattern: str=r"prc: hpc trc: (\d+) gen: (\d+) pop: (\d+) ftol: ([\deE\.-]+) time: ([\d\.]+) results: (\[.*\])") -> dict[tuple[str,str,str], any]:
    with open(file_name, "r") as f:
        log_lines = f.readlines()

    data = defaultdict(lambda: {"results":[], "generations": 0, "time": 0, "count": 0})
    
    pattern = re.compile(raw_pattern)
    
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

    # considered_results = []
    # for key, results in data.items():
    #     if key[0] == 400 and key[1] in [50]:
    #         considered_results.append((key[2], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2), results["results"]))

    # for c in considered_results:
    #     print(c, end="\n\n")

    return data

N = 8
G = 225

def parse_log_1(file_name: str, raw_pattern: str=r"prc: hpc trc: (\d+) gen: (\d+) pop: (\d+) ftol: ([\deE\.-]+) time: ([\d\.]+) first results: (\[.*\]) last results: (\[.*\])") -> dict[tuple[str,str,str], any]:
    with open(file_name, "r") as f:
        log_lines = f.readlines()

    data = defaultdict(lambda: {"first_results":[], "last_results":[], "generations": 0, "time": 0, "count": 0})
    
    pattern = re.compile(raw_pattern)
    
    for line in log_lines:
        match = pattern.match(line)
        if match:
            trc = int(match.group(1))
            gen = int(match.group(2))
            pop = int(match.group(3))
            ftol = float(match.group(4))
            time = float(match.group(5))
            first_results = eval(match.group(6))
            last_results = eval(match.group(7))
            
            key = (trc, pop, ftol)
            # print(key, gen, len(last_results))
            if key not in data.keys() or len(data[key]["last_results"]) < len(last_results):
            # if key[0] == 5000 and gen == G and len(last_results) == N:
                data[key]["first_results"] = first_results
                data[key]["last_results"] = last_results
            data[key]["generations"] += gen
            data[key]["time"] += time
            data[key]["count"] += 1

    return data

def normalize_data_0(data: dict[tuple[str,str,str], any]) -> None:
    for key, value in data.items():
        trc = key[0]
        value['results'] = [(r[0]/trc, r[1]/trc) for r in value['results']]

def normalize_data_1(data: dict[tuple[str,str,str], any]) -> None:
    for key, value in data.items():
        trc = key[0]
        value['first_results'] = [(r[0]/trc, r[1]/trc) for r in value['first_results']]
        value['last_results'] = [(r[0]/trc, r[1]/trc) for r in value['last_results']]

def plot_results_0(parsed_data: dict, folder: str):
    for key, data in parsed_data.items():
        file_name = f"{folder}parsed_data_{key}"
        x, y = zip(*data["results"])
        count = data["count"]
        gen = round(data["generations"] / count, 2)
        time = round(data["time"] / count, 2)
        legend_text = f"mean generation: {gen}\nmean time: {time} s"

        plt.figure(figsize=(12, 9))
        plt.scatter(x, y, s=100, color="red")
        # for i in range(len(x)):
        #     plt.plot([x[i], x[i]], [min(y), y[i]], color='blue')
        # for i in range(len(y)):
        #     plt.plot([min(x), x[i]], [y[i], y[i]], color='green')
        # plt.title(f"Optimization Results trc: {key[0]}, pop: {key[1]}, ftol: {key[2]}")
        plt.xlabel("Duration", fontsize=30)
        plt.ylabel("Cost", fontsize=30)
        plt.tight_layout()
        plt.legend([legend_text], loc="upper right", fontsize=10, frameon=True)
        plt.savefig(file_name + ".svg")
        plt.close()

def mean (x):
    return round(sum(x)/len(x),2)

def plot_results_1(parsed_data: dict, folder: str):
    for key, data in parsed_data.items():

        file_name = f"{folder}parsed_data_{key[0]}_baseline"
        # file_name = f"{folder}parsed_data_{key}_rnd_{N}_{G}"
        fx, fy = zip(*data["first_results"])
        lx, ly = zip(*data["last_results"])
        count = data["count"]
        time = round(data["time"] / count, 2)
        # print(key[0], "\t&", time, "s \t\t&", mean(fx), "\t&", mean(lx), "\t&", round(mean(lx)/mean(fx)*100-100,2), "% \t\t&", mean(fy), "\t&", mean(ly), "\t&", round(mean(ly)/mean(fy)*100-100,2), "% \\\\")
        print(key[0], "\t&", round(min(fx),2), "\t&", round(min(lx),2), "\t&", round(min(lx)/min(fx)*100-100,2))
        print(key[0], "\t&", round(min(fy),2), "\t&", round(min(ly),2), "\t&", round(min(ly)/min(fy)*100-100,2))
        plt.figure(figsize=(12, 9))
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(fx, fy, s=100, color="red", label="baseline")
        plt.scatter(lx, ly, s=100, color="blue", label="optimized")
        # plt.title(f"Optimization Results trc: {key[0]}, pop: {key[1]}, ftol: {key[2]}")
        plt.xlabel("Duration", fontsize=30)
        plt.ylabel("Cost", fontsize=30)
        plt.tight_layout()
        plt.legend(fontsize=20, frameon=True)
        plt.savefig(file_name + ".svg")
        plt.close()

def log_func(x, a, b):
    return a * np.log(x) + b

def plot_trc(data: dict, folder: str):
    considered_results = []
    for key, results in data.items():
        # if key[1] == 50 and key[2] == 2.5e-05 and key[0] in [1000, 5000, 10000]:
        if key[1] == 50 and key[2] == 2.5e-05 and key[0] != 500:
            considered_results.append((key[0], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2)))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    trc, gen, time = zip(*sorted_considered_results)

    trc = np.array(trc)
    time = np.array(time)
    # print("trc: ", trc)
    # print("time_trc: ", [int(s) for s in time])
    # print("h: ", [int(s /3600) for s in time])
    # print("min: ", [int((s %3600)/60) for s in time])
    # print("s: ", [(int(s %3600)%60) for s in time])

    plt.figure(figsize=(12, 9))
    plt.yscale('log')
    # plt.scatter(trc, time, color="red", label="time [s]")
    plt.plot(trc, time, color="red", label="time [s]")
    try:
        popt, _ = curve_fit(log_func, trc, time)
        trc_fit = np.linspace(min(trc), max(trc), 300)
        time_fit = log_func(trc_fit, *popt)
        plt.plot(trc_fit, time_fit, color="darkred", linestyle="--", label="trend (time)")
    except Exception as e:
        print("Could not fit logarithmic trend:", e)

    # plt.title(f"Analysis of pop: 50 ftol: 2.5e-05")
    plt.xlabel("traces", fontsize=30)
    plt.ylabel("time", fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    # plt.xticks(trc, rotation=60)
    plt.grid(False)
    plt.tight_layout()
    plt.legend(fontsize=30)
    plt.savefig(folder + "res_trc_time.svg")
    plt.close()

    considered_results = []
    for key, results in data.items():
        # if key[1] == 50 and key[2] == 2.5e-05 and key[0] in [1000, 5000, 10000]:
        if key[1] == 50 and key[2] == 2.5e-05 and key[0] != 500:
            considered_results.append((key[0], results["results"]))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    trc, res = zip(*sorted_considered_results)
    colors = ['#6ba646', '#91bc62', '#bbd688', '#e5eab3', '#f5d37d', '#f3ab62', '#ef7f47', '#ea5533']
    # colors = ['#6ba646', '#f5d37d', '#ea5533']

    plt.figure(figsize=(12, 9))
    plt.yscale('log')
    plt.xscale('log')

    for i, sol in enumerate(res):
        x, y = zip(*sol)
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x, y, s=100, color=colors[int(i/len(res)*len(colors))], label=f"trc {trc[i]}")

    plt.xlabel("Duration", fontsize=30)
    plt.ylabel("Cost", fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.grid(False)
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig(f"{folder}res_trc_res.svg")
    plt.close()

def plot_pop(data, folder: str, trc: int):
    considered_results = []
    for key, results in data.items():
        if key[0] == trc and key[2] == 2.5e-05:
            considered_results.append((key[1], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2), results["results"]))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    # colors = [(1, 0, 0, alpha) for alpha in np.linspace(0.3, 1, len(considered_results))]
    # colors = ['yellow', 'orange', 'red', 'green', 'blue', 'purple', 'black', 'cyan']
    colors = ['#6ba646', '#bbd688', '#f5d37d', '#f3ab62', '#ea5533']
    pop, gen, time, res = zip(*sorted_considered_results)

    plt.figure(figsize=(12, 9))
    # plt.yscale('log')
    # plt.plot(pop, gen, color="blue", label="gen")
    plt.plot(pop, time, color="red", label="time [s]")
    # plt.title(f"Analysis of trc: 400 ftol: 2.5e-05")
    plt.xlabel("pop size", fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.xticks(pop)
    plt.grid(False)
    plt.tight_layout()
    plt.legend(fontsize=30)
    plt.savefig(f"{folder}res_pop_time_{trc}.svg")
    plt.close()

    plt.figure(figsize=(12, 9))
    for i, sol in enumerate(res):
        x, y = zip(*sol)
        x = np.array(x)
        y = np.array(y)

        plt.scatter(x, y, s=100, color=colors[i], label=f"pop {pop[i]}")

        if len(x) >= 3:
            coeffs = np.polyfit(x, y, deg=1)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = poly(x_fit)
            # plt.plot(x_fit, y_fit, color=colors[i], linestyle='--')
    # plt.title(f"Analysis of trc: 400 ftol: 2.5e-05")
    plt.xlabel("Duration", fontsize=30)
    plt.ylabel("Cost", fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.grid(False)
    plt.tight_layout()
    plt.legend(fontsize=30)
    plt.savefig(f"{folder}res_pop_res_{trc}.svg")
    plt.close()

def plot_ftol(data, folder: str, trc: int):
    considered_results = []
    for key, results in data.items():
        if key[0] == trc and key[1] == 50 and key[2] not in [2.5e-09, 2.5e-08, 2.5e-07]:
            considered_results.append((key[2], round(results["generations"]/results["count"], 2), round(results["time"]/results["count"], 2), results["results"]))

    sorted_considered_results = sorted(considered_results, key=lambda p: p[0])

    # colors = [(1, 0, 0, alpha) for alpha in np.linspace(0.3, 1, len(considered_results))]
    # colors = ['yellow', 'orange', 'red', 'green', 'blue', 'purple', 'black', 'cyan']
    # colors = ['#6ba646', '#91bc62', '#bbd688', '#e5eab3', '#f5d37d', '#f3ab62', '#ef7f47', '#ea5533']
    colors = ['#6ba646', '#bbd688', '#e5eab3', '#f3ab62', '#ea5533']
    ftol, gen, time, res = zip(*sorted_considered_results)
    tick_labels = [f"{x:.1e}" for x in ftol] 
    print(ftol)

    plt.figure(figsize=(12, 9))
    # plt.yscale('log')
    plt.xscale('log')
    # plt.plot(ftol, gen, color="blue", label="gen")
    plt.plot(ftol, time, color="red", label="time [s]")
    # plt.title(f"Analysis of trc: 400 pop: 50")
    plt.xlabel("ftol", fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.xticks(ftol, tick_labels, rotation=45, ha='right')  # Set ticks and labels
    plt.tight_layout()
    plt.grid(False)
    plt.legend(fontsize=30)
    plt.savefig(f"{folder}res_ftol_time_{trc}.svg")
    plt.close()
    
    plt.figure(figsize=(12, 9))
    # plt.yscale('log')
    # plt.xscale('log')
    for i, sol in enumerate(res):
        x, y = zip(*sol)
        x = np.array(x)
        y = np.array(y)

        plt.scatter(x, y, s=100, color=colors[i], label=f"ftol {ftol[i]}")

        if len(x) >= 3:
            coeffs = np.polyfit(x, y, deg=1)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = poly(x_fit)
            # plt.plot(x_fit, y_fit, color=colors[i], linestyle='--')

    # plt.title(f"Analysis of trc: 400 pop: 50")
    plt.xlabel("Duration", fontsize=30)
    plt.ylabel("Cost", fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.grid(False)
    plt.tight_layout()
    plt.legend(fontsize=30)
    plt.savefig(f"{folder}res_ftol_res_{trc}.svg")
    plt.close()

def parse_plot_0(log_file_0: str, folder: str):
    parsed_data_0 = parse_log_0(log_file_0)
    normalize_data_0(parsed_data_0)
    # plot_results_0(parsed_data_0, folder)
    plot_trc(parsed_data_0, folder)
    plot_pop(parsed_data_0, folder, 400)
    # plot_pop(parsed_data_0, folder, 500)
    plot_ftol(parsed_data_0, folder, 400)
    # plot_ftol(parsed_data_0, folder, 500)

def parse_plot_1(log_file_1: str, folder: str):
    raw_pattern = r"prc: hpc trc: (\d+) gen: (\d+) pop: (\d+) ftol: ([\deE\.-]+) time: ([\d\.]+) first results: (\[.*\]) first gene: (\[.*\]) last results: (\[.*\]) last gene: (\[.*\])"
    raw_pattern = r"prc: hpc trc: (\d+) gen: (\d+) pop: (\d+) ftol: ([\deE\.-]+) time: ([\d\.]+) first results: (\[.*?\]) last results: (\[.*?\])"
    parsed_data_1 = parse_log_1(log_file_1)
    normalize_data_1(parsed_data_1)
    plot_results_1(parsed_data_1, folder)


if __name__ == "__main__":
    log_file_0 = "ignored/sim_8.txt"
    log_file_1 = "ignored/sim_9.txt"
    log_file_2 = "ignored/sim_consulta_0.txt"
    folder_0 = "ignored/parsed_res/"
    folder_1 = "ignored/parsed_res_consulta/"


    # parse_log_0(log_file_0)
    # parse_plot_0(log_file_0, folder_0)
    # parse_plot_1(log_file_1, folder_0)
    parse_plot_1(log_file_2, folder_1)

    