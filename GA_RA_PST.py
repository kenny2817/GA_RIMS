from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import sys
import numpy as np                              # type: ignore
import matplotlib.pyplot as plt                 # type: ignore
from typing import Dict, List
from scipy.stats import trim_mean               # type: ignore

from Process import Process
from Bpmn import Bpmn
from RIMS_tool.core.run_simulation import run_simulation

from pymoo.optimize import minimize             # type: ignore
from pymoo.termination import get_termination   # type: ignore
from pymoo.algorithms.moo.nsga2 import NSGA2    # type: ignore
from pymoo.core.problem import Problem          # type: ignore
from pymoo.core.sampling import Sampling        # type: ignore
from pymoo.core.mutation import Mutation        # type: ignore
from pymoo.core.crossover import Crossover      # type: ignore
from pymoo.config import Config                 # type: ignore
Config.warnings['not_compiled'] = False


class GA_RA_PST_Problem(Problem):

    def __init__(
            self,
            paths: Dict[str, str],
            bpmn: Bpmn,
            process: Process,
            number_traces: int = 1,
            number_simulations: int = 1,
            number_processes: int = 1,
            mutation_treshold: float = 0.1,
        ):

        if number_traces < 1:
            raise ValueError('number of traces should be equal or grater than 1')
        if number_processes < 1:
            raise ValueError('number of threads should be equal or grater than 1')
        if number_simulations < 1:
            raise ValueError('number of simulations should be equal or greater than 1')
        if not (0 <= mutation_treshold <= 1):
            raise ValueError('mutation treshold should be [0,1]')
        
        self.number_traces = number_traces
        self.number_simulations = number_simulations
        self.number_processes = number_processes
        self.mutation_threshold = mutation_treshold
        
        self.decsion_tasks, self.upper_bound_values = process.gene_upper_bound()

        length_gene = len(self.upper_bound_values) * number_traces
        if (length_gene < 1):
            raise ValueError('the lengh of the gene must be greater that 0 to have some optimization')

        args = {
            "n_var": length_gene,                           # variable for each gene
            "n_obj": 2,                                     # time, cost
            "n_constr": 0,                                  # no constraint
            "xl": [1] * length_gene,                        # lower bound 0 is the base bpmn, [1,xu] are the genetic choices
            "xu": self.upper_bound_values * number_traces   # upper bound (included)
        }
        super().__init__(**args)

        self.paths = paths
        self.bpmn = bpmn
        self.process = process

    def _simulate(self, x: list):
        paths = self.paths
        n = int(self.n_simulations / self.n_process)
        r = int(self.n_simulations % self.n_process)

        with open(paths["redirect"], "w") as file:
            # Save
            original_stderr = sys.stderr
            original_stdout = sys.stdout
            try:
                # Redirect
                sys.stderr = file
                sys.stdout = file
                with ProcessPoolExecutor(max_workers=self.n_process) as executor:
                    futures = []
                    
                    for thread_id in range(self.n_process):
                        cleanup_directory(paths["output_folder"] + "_thread_" + str(thread_id))
                        future = executor.submit(run_simulation, paths["petrinet_file"], paths["simulation_params"], x, n + (1 if thread_id < r else 0), self.num_traces, paths["output_folder_name"] + "_thread_" + str(thread_id))
                        futures.append(future)
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
    def _do(self, problem: GA_RA_PST_Problem, X: List, **kwargs):
        for i in range(len(X)):
            if np.random.rand() < problem.mutation_treshold:
                X[i] = np.random.randint(problem.xl, problem.xu)
        return X

class CustomCrossover(Crossover):
    def __init__(self, n_parents: int = 2, n_offsprings: int = 2):
        super().__init__(n_parents, n_offsprings)
    
    def _do(self, problem: GA_RA_PST_Problem, X: List, **kwargs):
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

def debug_process(process: Process):

    print("\nRoles: ", [role.get_name() for role in process.get_roles().values()])

    print("\nTasks:")
    for task in process.get_tasks().values():
        print(f"{task.get_name()} can be done by {task.get_roles()}")

    print("\nCore tasks:", [task.get_name() for task in process.get_tasks().values() if task.get_iscore()])
    
    print("\nDecision trees:\n")
    for task in process.tasks.values():
        print(f"{task.get_name()} got {task.get_leaves()} leaves")
        task.print_decision_tree()

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

def final_cleanup(paths: dict[str: str], number_process: int = 1):
    txt_file = paths["redirect"]
    if os.path.exists(txt_file) and os.path.isfile(txt_file):
            os.remove(txt_file)
    
    for suffix in range(number_process):
        folder = paths["output_folder"] + "_thread_" + str(suffix)
        if os.path.exists(folder):
            shutil.rmtree(folder)

if __name__ == "__main__":

    diagram_name = "diagram_4"
    diagram_folder_file = f"./{diagram_name}/{diagram_name}"
    output_folder = f"./output/output_{diagram_name}"
    paths = {
        "output_folder": output_folder,
        "results": f"{output_folder}/results",
        "progression": f"{output_folder}/progession",
        "redirect": f"{output_folder}/redirect.txt",
        "output_folder_name": diagram_name,
        "diagram_folder_file": diagram_folder_file,
        "bpmn_file": f"{diagram_folder_file}.bpmn",
        "petrinet_file": f"{diagram_folder_file}.pnml",
        "simulation_params": f"{diagram_folder_file}.json"
    }

    process = Process(paths["simulation_params"])
    bpmn = Bpmn(paths["bpmn_file"])
    bpmn.set_core_tasks(process.get_tasks())
    process.compute_decision_tree()

    # debug_process(process)

    population_size = 20
    n_gen = 30
    args = {
        "paths": paths,
        "bpmn": bpmn,
        "process": process,
        "number_traces": 30,
        "number_simulations": 100,
        "number_processes": 5,
        "mutation_treshold": 0.1,
    }

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=IntegerRandomSampling(),
        crossover=CustomCrossover(),
        mutation=CustomMutation(),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", n_gen)
    
    problem = GA_RA_PST_Problem(**args)
    
    cleanup_directory(paths["output_folder"])

    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=False,
        save_history=True
    )

    plot_history(res, paths["progression"])
    plot_results(res.F, paths["results"])

    final_cleanup(paths)