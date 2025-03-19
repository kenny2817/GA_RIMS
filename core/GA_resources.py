import numpy as np                              # type: ignore
import os
import shutil
import threading
import sys
import json
import matplotlib.pyplot as plt                 # type: ignore
import time

from pymoo.optimize import minimize             # type: ignore
from pymoo.termination import get_termination   # type: ignore
from pymoo.algorithms.moo.nsga2 import NSGA2    # type: ignore
from pymoo.core.problem import Problem          # type: ignore
from pymoo.core.sampling import Sampling        # type: ignore
from pymoo.core.mutation import Mutation        # type: ignore
from pymoo.core.crossover import Crossover      # type: ignore

from concurrent.futures import ProcessPoolExecutor

from scipy.stats import trim_mean               # type: ignore

from RIMS_tool.core.run_simulation import run_simulation

from pymoo.config import Config                 # type: ignore
Config.warnings['not_compiled'] = False

class ResourceAssignmentProblem(Problem):
    def __init__(self, map_decision_activity: list, num_traces: int, paths: dict[str: str], mutation_treshold: float, n_simulations: int, n_process: int = 1):
        if num_traces < 1:
            raise ValueError('num_traces should be grater than 0')
        elif mutation_treshold < 0 or mutation_treshold > 1:
            raise ValueError('mutation_treshold should be [0,1]')
        elif n_process < 1:
            raise ValueError('number of threads should be equal or grater that 1')
        else:
            n_var = len(map_decision_activity) * num_traces
            n_obj = 2                                   # cost and time
            lower_bound = [0] * n_var
            upper_bound = [(len(mapping) -1) for mapping in map_decision_activity] * num_traces
            
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=lower_bound, xu=upper_bound)
            
            self.mutation_treshold = mutation_treshold
            self.paths = paths
            self.map_decision_activity = map_decision_activity
            self.num_traces = num_traces
            self.n_simulations = n_simulations
            self.n_process = n_process
    
    def _decode(self, x: list):
        resource_assignment = []
        trace_length = len(self.map_decision_activity)

        # Process each trace
        for trace_idx in range(self.num_traces):
            start_idx = trace_idx * trace_length
            
            # Decode one trace
            trace_assignment = []
            for i, decision in enumerate(self.map_decision_activity):
                decision_index = int(x[start_idx + i])
                trace_assignment.append(decision[decision_index])
            
            resource_assignment.append(trace_assignment)
        
        return resource_assignment
    
    def _simulate(self, x: list):
        paths = self.paths
        n = int(self.n_simulations / self.n_process)
        r = int(self.n_simulations % self.n_process)

        with open(paths["redirect"], "w") as file:
            # Save
            original_stderr = sys.stderr
            original_stdout = sys.stdout
            # Redirect
            sys.stderr = file
            sys.stdout = file
            try:
                with ProcessPoolExecutor(max_workers=self.n_process) as executor:
                    futures = []
                    
                    for thread_id in range(self.n_process):
                        cleanup_directory(paths["output_folder"] + "_thread_" + str(thread_id))
                        future = executor.submit(run_simulation, paths["petrinet_file"], paths["simulation_params"], x, n + (1 if thread_id < r else 0), self.num_traces, paths["output_folder_name"] + "_thread_" + str(thread_id))
                        futures.append(future)
                # serial
                # result = run_simulation(paths["petrinet_file"], paths["simulation_params"], x, n, self.num_traces, paths["output_folder_name"])
            finally:
                # Restore
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        result = []
        for future in futures:
            result += future.result() 

        duration, cost = zip(*result)

        duration = trim_mean(duration, proportiontocut=0.025)
        cost = trim_mean(cost, proportiontocut=0.025)
        
        return [duration, cost]

    def _evaluate(self, X: list, out, *args, **kwargs):        
        # print("|", end="")
        # sys.stdout.flush()
        F = []
        for x in X:
            res = self._simulate(x)
            
            F.append(res)

        out["F"] = np.array(F)

class IntegerRandomSampling(Sampling):
    def _do(self, problem, n_samples: int, **kwargs):
        return np.random.randint(
            problem.xl,
            problem.xu + 1,
            size=(n_samples, problem.n_var)
        )

class CustomMutation(Mutation):
    def _do(self, problem: ResourceAssignmentProblem, X: list, **kwargs):
        for i in range(len(X)):
            if np.random.rand() < problem.mutation_treshold:
                X[i] = np.random.randint(problem.xl, problem.xu)
        return X

class CustomCrossover(Crossover):
    def __init__(self, n_parents: int = 2, n_offsprings: int = 2):
        super().__init__(n_parents, n_offsprings)
    
    def _do(self, problem: ResourceAssignmentProblem, X: list, **kwargs):
        n_matings = X.shape[1]
        n_var = X.shape[2]
        offsprings = np.empty((self.n_offsprings, n_matings, n_var))

        for k in range(n_matings):
            parent1, parent2 = X[0, k], X[1, k]
            crossover_point = np.random.randint(1, n_var)
            offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            offsprings[0, k, :] = offspring1
            offsprings[1, k, :] = offspring2
        
        return offsprings

def plot_history(result, file_name: str, offset: int = 0):
    history = [algo.pop.get("F") for algo in result.history]

    min_duration = []
    max_duration = []
    min_cost = []
    max_cost = []
    for i, gen in enumerate(history): 
        if i >= offset:
            min_duration.append(np.min(gen[:,0]))
            max_duration.append(np.max(gen[:,0]))
            min_cost.append(np.min(gen[:,1]))
            max_cost.append(np.max(gen[:,1]))

    plt.figure(figsize=(12, 7))
    plt.yscale('log', base=10)
    plt.plot(min_duration, label="Best duration", color='#90EE90')
    plt.plot(max_duration, label="Worst duration", color='#006400')
    plt.plot(min_cost, label="Best cost", color="#ADD8E6")
    plt.plot(max_cost, label="worst cost", color="#00008B")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.title("Objective Value Progression")
    plt.legend()
    plt.savefig(file_name + ".png")
    plt.close()

def plot_results(solutions: list, file_name: str):
    x, y = zip(*solutions)

    plt.figure(figsize=(12, 7))
    plt.scatter(x, y, color='r')
    plt.title("Best solutions")
    plt.xlabel("duration")
    plt.ylabel("cost")
    plt.grid(True)
    plt.savefig(file_name + ".png")
    plt.close('all')

def cleanup_directory(directory_path: str):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)

        os.makedirs(directory_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def decision_activity_init(path: dict[str: str]) -> list:
    map_decision_activity = []
    with open(path) as file:
        data = json.load(file)
        for elem in data['tasks'].values():
            role = elem['role']
            if isinstance(role, list):  # list assert
                map_decision_activity.append(role)
    return map_decision_activity

def final_cleanup(paths: dict[str: str], n_process: int):
    txt_file = paths["redirect"]
    if os.path.exists(txt_file) and os.path.isfile(txt_file):
            os.remove(txt_file)
    
    for suffix in range(n_process):
        folder = paths["output_folder"] + "_thread_" + str(suffix)
        if os.path.exists(folder):
            shutil.rmtree(folder)

if __name__ == "__main__":

    diagram_name = "diagram_2"

    paths = {
        "progression": f"./output/output_{diagram_name}/progession",
        "results": f"./output/output_{diagram_name}/results",
        "output_folder_name": diagram_name,
        "output_folder": f"./output/output_{diagram_name}",
        "petrinet_file": f"./{diagram_name}/{diagram_name}.pnml",
        "simulation_params": f"./{diagram_name}/{diagram_name}.json",
        "redirect": "./redirect.txt"
    }

    algorithm = NSGA2(
        pop_size=20,
        sampling=IntegerRandomSampling(),
        crossover=CustomCrossover(),
        mutation=CustomMutation(),
        eliminate_duplicates=True
    )

    map_decision_activity = decision_activity_init(paths["simulation_params"])

    num_traces = 30
    mutation_treshold = 0.1
    n_simulations = 100
    n_gen = 30
    n_sim = 5
    termination = get_termination("n_gen", n_gen)

    cleanup_directory(paths["output_folder"])

    for n_process in range(5,8):
        mean = 0
        for sim in range(n_sim):
            start_time = time.time()


            problem = ResourceAssignmentProblem(
                map_decision_activity, 
                num_traces, 
                paths, 
                mutation_treshold, 
                n_simulations,
                n_process
            )

            # for _ in range(n_gen):
            #     print("|", end="")
            # print("")

            res = minimize(
                problem,
                algorithm,
                termination,
                verbose=False,
                save_history=True
            )

            # print("")

            plot_history(res, paths["progression"] + f"_{n_process}_{sim}")
            plot_results(res.F, paths["results"] + f"_{n_process}_{sim}")


            end_time = time.time()
            execution_time = end_time - start_time
            print(f"nth: {n_process} Execution time: {execution_time:.4f} seconds")

            mean += execution_time
        print(f"nth: {n_process} mean: {mean / n_sim:.4f}")
        final_cleanup(paths, n_process)
    