from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import sys
import numpy as np                              # type: ignore
import matplotlib.pyplot as plt                 # type: ignore
from typing import Dict, List
from scipy.stats import trim_mean               # type: ignore

from PetriNet import PetriNet
from Parameters import Parameters
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
            # process: Process,
            number_traces: int = 1,
            number_simulations: int = 1,
            number_processes: int = 1,
            mutation_threshold: float = 0.1,
            mutation_proportion: float = 0.01,
        ):

        if number_traces < 1:
            raise ValueError('number of traces should be equal or grater than 1')
        if number_processes < 1:
            raise ValueError('number of threads should be equal or grater than 1')
        if number_simulations < 1:
            raise ValueError('number of simulations should be equal or greater than 1')
        if not (0 <= mutation_threshold <= 1):
            raise ValueError('mutation treshold should be [0,1]')
        
        self.number_traces = number_traces
        self.number_simulations = number_simulations
        self.number_processes = number_processes
        self.mutation_threshold = mutation_threshold
        self.mutation_proportion = mutation_proportion
        
        self.upper_bound_values = bpmn.get_upper_bound()

        self.length_gene = len(self.upper_bound_values) * number_traces
        if (self.length_gene < 1):
            raise ValueError('the lengh of the gene must be greater that 0 to have some optimization')
        
        self.length_mutation = int(self.length_gene * mutation_proportion)
        if self.length_mutation < 1: self.length_mutation = 1

        args = {
            "n_var": self.length_gene,                           # variable for each genoma
            "n_obj": 2,                                     # time, cost
            "n_constr": 0,                                  # no constraint
            "xl": [0] * self.length_gene,                        # lower bound 0 is the base bpmn, [1,xu] are the genetic choices
            "xu": self.upper_bound_values * number_traces   # upper bound (included)
        }
        super().__init__(**args)

        self.paths = paths
        self.bpmn = bpmn
        # self.process = process

        # self.gen = 0

    def _simulate(self, x: List):
        paths = self.paths
        n = int(self.number_simulations // self.number_processes)
        r = int(self.number_simulations % self.number_processes)

        with open(paths["redirect"], "w") as file:
            # Save
            original_stderr = sys.stderr
            original_stdout = sys.stdout
            try:
                # Redirect
                sys.stderr = file
                sys.stdout = file
                with ProcessPoolExecutor(max_workers=self.number_processes) as executor:
                    params = {
                        "PATH_PETRINET": paths["petrinet_file"],
                        "PATH_PARAMETERS": paths["simulation_params"],
                        "GENE": x,
                        "N_TRACES": self.number_traces
                    }
                    futures = []
                    
                    for thread_id in range(self.number_processes):
                        params["N_SIMULATION"] = n + (1 if thread_id < r else 0)
                        params["NAME"] = paths["diagram_name"] + "_thread_" + str(thread_id)
                        cleanup_directory(paths["output_folder"] + "_thread_" + str(thread_id))
                        future = executor.submit(run_simulation, **params)
                        futures.append(future)
            finally:
                # Restore
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        result = []
        for future in futures:
            result += future.result() 

        duration, cost = zip(*result)

        proportiontocut = 0.025
        duration = trim_mean(duration, proportiontocut=proportiontocut)
        cost = trim_mean(cost, proportiontocut=proportiontocut)
        
        return [duration, cost]

    def _evaluate(self, X: list, out, *args, **kwargs):
        paths = self.paths
        population_size = X.shape[0]
        proportion_to_cut = 0.025
        n = population_size // self.number_processes
        r = population_size % self.number_processes

        params = {
            "PATH_PETRINET": paths["petrinet_file"],
            "PATH_PARAMETERS": paths["simulation_params"],
            "N_TRACES": self.number_traces,
            "N_SIMULATION": self.number_simulations
        }

        futures = []
        F = []
        with ProcessPoolExecutor(max_workers=self.number_processes) as executor:
            for index in range(population_size):
                params["GENE"] = X[index]
                params["NAME"] = paths["diagram_name"] + f"_index_{index}"
                cleanup_directory(paths["output_folder"] + f"_index_{index}")
                future = executor.submit(run_simulation, **params)
                futures.append((index, future))
            
            for index, future in sorted(futures, key=lambda x: x[0]):
                result = future.result()
                duration, cost = zip(*result)
                duration = trim_mean(duration, proportiontocut=proportion_to_cut)
                cost = trim_mean(cost, proportiontocut=proportion_to_cut)
                F.append([duration, cost])

        out["F"] = np.array(F)

        # plot_results(out["F"], self.paths["pareto"] + f"_gen_{self.gen}")
        # self.gen += 1

class IntegerRandomSampling(Sampling):
    def _do(self, problem, n_samples: int, **kwargs):
        return np.random.randint(
            problem.xl,
            problem.xu + 1,
            size=(n_samples, problem.n_var)
        )
class CustomMutation(Mutation):
    # def _do(self, problem, X, **kwargs):
    #     n, m = X.shape
    #     for i in range(n):
    #         if np.random.rand() < problem.mutation_threshold:
    #             j = np.random.randint(0, m)
    #             old_val = X[i, j]
    #             bounds = int(problem.xl[j]), int(problem.xu[j]) +1
    #             new_val = np.random.randint(*bounds)
    #             while new_val == old_val:
    #                 new_val = np.random.randint(*bounds)
    #             X[i, j] = new_val
    #     return X
    
    def _do(self, problem, X, **kwargs):
        n, m = X.shape
        for i in range(n):
            if np.random.rand() < problem.mutation_threshold:
                for _ in range(problem.length_mutation):
                    j = np.random.randint(0, m)
                    old_val = X[i, j]
                    bounds = int(problem.xl[j]), int(problem.xu[j]) +1
                    new_val = np.random.randint(*bounds)
                    while new_val == old_val:
                        new_val = np.random.randint(*bounds)
                    X[i, j] = new_val
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
    plt.yscale('log')
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
def final_cleanup(paths: dict[str: str], population_size: int = 1):
    txt_file = paths["redirect"]
    if os.path.exists(txt_file) and os.path.isfile(txt_file):
            os.remove(txt_file)
    
    for suffix in range(population_size):
        folder = paths["output_folder"] + "_index_" + str(suffix)
        if os.path.exists(folder):
            shutil.rmtree(folder)

if __name__ == "__main__":
    diagram_name = "diagram_4_3"
    diagram_folder_file = f"./{diagram_name}/{diagram_name}"
    output_folder = f"./output/output_{diagram_name}"
    paths = {
        "diagram_name": diagram_name,
        "output_folder": output_folder,
        "results": f"{output_folder}/results",
        "progression": f"{output_folder}/progession",
        "pareto": f"{output_folder}/pareto",
        "redirect": f"{output_folder}/redirect.txt",
        "output_folder_name": diagram_name,
        "diagram_folder_file": diagram_folder_file,
        "bpmn_file": f"{diagram_folder_file}.bpmn",
        "petrinet_file": f"{diagram_folder_file}.pnml",
        "input_params": f"{diagram_folder_file}.json",
        "simulation_params": f"{output_folder}/simulation_parameters.json"
    }

    petrinet = PetriNet(paths["bpmn_file"])
    petrinet.save_net(paths["petrinet_file"])

    # process = Process(paths["input_params"])
    # process.compute_decision_tree()
    # process.debug()

    bpmn = Bpmn(paths["bpmn_file"])
    # bpmn.set_core_tasks(process.get_tasks())

    parameters = Parameters(paths["input_params"])
    # parameters.add_tasks(process.get_tasks())
    parameters.add_mapping(bpmn)
    parameters.save(paths["simulation_params"])

    population_size = 200
    # number_traces = 100
    number_simulations = 10
    # n_gen = 30
    number_processes = 10

    for n_gen in range(200, 201, 30):
        termination = get_termination("n_gen", n_gen)
        for number_traces in range(20, 101, 100):
            problem = GA_RA_PST_Problem(
                paths=paths,
                bpmn=bpmn,
                # process=process,
                number_traces=number_traces,
                number_simulations=number_simulations,
                number_processes=number_processes,
                mutation_threshold=0.1,
                mutation_proportion=0.1
            )

            algorithm = NSGA2(
                pop_size=population_size,
                sampling=IntegerRandomSampling(),
                crossover=CustomCrossover(),
                mutation=CustomMutation(),
                eliminate_duplicates=True
            )

            # cleanup_directory(paths["output_folder"])
            
            res = minimize(
                problem,
                algorithm,
                termination,
                verbose=True,
                save_history=True
            )

            with open("simulation_time_1.txt", "a") as file: 
                file.write(f"prc: {number_processes} trc: {number_traces} gen: {n_gen} pop: {population_size} time: {res.exec_time}\n")
            print(f"prc: {number_processes} trc: {number_traces} gen: {n_gen} pop: {population_size} time: {res.exec_time}")

            plot_history(res, paths["progression"] + f"_{number_processes}_{number_traces}")
            plot_results(res.F, paths["results"] + f"_{number_processes}_{number_traces}")
            # plot_pareto(res, paths["pareto"] + f"{number_processes}_{number_traces}")

    # plot_history(res, paths["progression"])
    # plot_results(res.F, paths["results"])

    final_cleanup(paths, population_size)
    # final_cleanup(paths, 100)

    # print(f"processes: {number_processes} time: {end_time - start_time}")
