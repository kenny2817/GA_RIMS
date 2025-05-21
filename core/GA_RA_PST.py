from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import sys
import numpy as np                              # type: ignore
from typing import Dict, List
from scipy.stats import trim_mean               # type: ignore

from petrinet import PetriNet
from parameters import Parameters
from bpmn import Bpmn
from RIMS_tool.core.run_simulation import run_simulation

from pymoo.optimize import minimize             # type: ignore
from pymoo.termination.default import DefaultMultiObjectiveTermination # type: ignore
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
            upper_bound: list[int],
            number_traces: int = 1,
            number_simulations: int = 1,
            mutation_threshold: float = 0.1,
            mutation_proportion: float = 0.01,
        ):

        if number_traces < 1:
            raise ValueError('number of traces should be equal or grater than 1')
        if number_simulations < 1:
            raise ValueError('number of simulations should be equal or greater than 1')
        if not (0 <= mutation_threshold <= 1):
            raise ValueError('mutation treshold should be [0,1]')
        
        self.number_traces = number_traces
        self.number_simulations = number_simulations
        self.mutation_threshold = mutation_threshold
        self.mutation_proportion = mutation_proportion
        

        length_gene = len(upper_bound) * number_traces
        if (length_gene < 1):
            raise ValueError('the lengh of the gene must be greater that 0 to have some optimization')
        
        self.length_mutation = int(length_gene * mutation_proportion)
        if self.length_mutation < 1: self.length_mutation = 1

        args = {
            "n_var": length_gene,                 # variable for each genoma
            "n_obj": 2,                           # time, cost
            "n_constr": 0,                        # no constraint
            "xl": [0] * length_gene,              # lower bound 0 is the base bpmn, [1,xu] are the genetic choices
            "xu": upper_bound * number_traces     # upper bound (included)
        }
        super().__init__(**args)

        self.paths = paths

    def _evaluate(self, X, out, *args, **kwargs):
        paths = self.paths
        population_size = X.shape[0]
        proportion_to_cut = 0.025

        params = {
            "PATH_PETRINET": paths["petrinet_file"],
            "PATH_PARAMETERS": paths["simulation_params"],
            "N_TRACES": self.number_traces,
            "N_SIMULATION": self.number_simulations
        }

        futures = []
        F = []
        with ProcessPoolExecutor() as executor:
            for index in range(population_size):
                # print(X[index])
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

class IntegerRandomSampling(Sampling):
    def _do(self, problem, n_samples: int, **kwargs):
        return np.random.randint(
            problem.xl,
            problem.xu + 1,
            size=(n_samples, problem.n_var)
        )
class CustomMutation(Mutation):
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

def cleanup_directory(directory_path: str):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)

        os.makedirs(directory_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        
def final_cleanup(paths: dict[str: str], population_size: int = 1):
    for suffix in range(population_size):
        folder = paths["output_folder"] + "_index_" + str(suffix)
        if os.path.exists(folder):
            shutil.rmtree(folder)

def estract_results(solutions) -> list[list[float]]:
    cost, duration = zip(*solutions)
    results = [[cost[i], duration[i]] for i in range(len(cost))]
    return results
    
def test():
    params = {
        "PATH_PETRINET": "t/test_0.pnml",
        "PATH_PARAMETERS": "t/test_0.json",
        "N_TRACES": 1,
        "N_SIMULATION": 1
    }
    params["GENE"] = []
    params["NAME"] = "palla"


    PetriNet(
        input_path="t/test_0.bpmn",
        output_path=params["PATH_PETRINET"]
    )

    run_simulation(**params)


if __name__ == "__main__":
    t = False
    if t:
        test()
    else:
        diagram_name = "diagram_5_0"
        diagram_folder_file = f"./{diagram_name}/{diagram_name}"
        output_folder = f"./output/output_{diagram_name}"
        paths = {
            "diagram_name": diagram_name,
            "output_folder": output_folder,
            "output_folder_name": diagram_name,
            "diagram_folder_file": diagram_folder_file,
            "bpmn_file": f"{diagram_folder_file}.bpmn",
            "petrinet_file": f"{diagram_folder_file}.pnml",
            "input_params": f"{diagram_folder_file}.json",
            "simulation_params": f"{output_folder}/simulation_parameters.json"
        }

        petrinet = PetriNet(
            input_path=paths["bpmn_file"],
            output_path=paths["petrinet_file"]
        )

        bpmn = Bpmn(
            input_path=paths["bpmn_file"]
        )

        parameters = Parameters(
            input_path=paths["input_params"],
            output_path=paths["simulation_params"],
            xor_mapping=bpmn.get_xor_mapping()
        )

        population_size = int(sys.argv[1])
        number_traces = int(sys.argv[2])
        plot_id = sys.argv[3]
        ftol = float(sys.argv[4])

        number_simulations = 10

        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=ftol,
            period=30,
            n_max_gen=10000,
            n_max_evals=1000000
        )

        problem = GA_RA_PST_Problem(
            paths=paths,
            upper_bound=parameters.get_upper_bound(),
            number_traces=number_traces,
            number_simulations=number_simulations,
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
        
        res = minimize(
            problem,
            algorithm,
            termination,
            verbose=True,
            save_history=True
        )

        if res.history:
            n_gen = len(res.history)
            last_execution_solutions = res.F
            last_execution_gene = res.X
            last_execution_results = estract_results(last_execution_solutions)

            first_execution_solutions = res.history[0].pop.get("F")
            first_execution_gene = res.history[0].get["X"]
            first_execution_results = estract_results(first_execution_solutions)
            with open("simulation_time.txt", "a") as file: 
                file.write(f"prc: hpc trc: {number_traces} gen: {n_gen} pop: {population_size} ftol: {ftol} time: {res.exec_time} first results: {first_execution_results} first gene: {first_execution_gene} last results: {last_execution_results} last gene: {last_execution_gene}\n")
        else:
            raise ValueError("there is no history")
        
        final_cleanup(paths, population_size)


