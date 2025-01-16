import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover

from RIMS_tool.core.run_simulation import run_simulation

# from pymoo.config import Config
# Config.warnings['not_compiled'] = False

def simulate(paths: dict[str:str], n_simulations: int):
    with open("redirect.txt", "w") as file:
        # Save
        original_stderr = sys.stderr
        original_stdout = sys.stdout
        # Redirect
        sys.stderr = file
        sys.stdout = file
        try:
            run_simulation(paths["petrinet_file"], paths["simulation_params"], paths["jsonl"], n_simulations, num_traces, paths["output_folder_name"])
        finally:
            # Restore
            sys.stdout = original_stdout
            sys.stderr = original_stderr

class ResourceAssignmentProblem(Problem):
    def __init__(self, map_decision_activity: list, num_traces: int, paths: dict[str: str], mutation_treshold: float, n_simulations: int):
        if num_traces < 1:
            raise ValueError('num_traces should be grater than 0')
        elif mutation_treshold < 0 or mutation_treshold > 1:
            raise ValueError('mutation_treshold should be [0,1]')
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
            self.iteration = 0
    
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

    def _fetch_results(self):
        L = []
        n = self.n_simulations
        jsonl = self.paths["jsonl"]

        with open(jsonl, 'rb') as file:
            file.seek(0, 2)
            end = file.tell()
            while len(L) <= n and end > 0:
                file.seek(end - 1024 if end > 1024 else 0, 0)
                L = file.read(end - file.tell()).splitlines()
                end = file.tell()

        duration = 0
        cost = 0

        for line in L[-n:]:
            line = line.decode()
            line = json.loads(line)
            results = line['results']
            duration += float(results[0])
            cost += float(results[1])

        return [duration / n, cost / n]

    def _evaluate(self, X: list, out, *args, **kwargs):
        jsonl = self.paths["jsonl"]
        output = { "iteration": self.iteration }
        
        with open(jsonl, "a") as json_file:
            json_file.write(json.dumps(output) + "\n")

        F = []
        solutions = []
        for i, x in enumerate(X):
            solutions.append({
                "solution_id": i,
                "genes": x.tolist(),
                "resource_assignment": self._decode(x)
            })

            with open(jsonl, "a") as json_file:
                json_file.write(json.dumps(solutions[-1]) + "\n")

            simulate(self.paths, self.n_simulations)
            
            F.append(self._fetch_results())

        out["F"] = np.array(F)
        self.iteration += 1

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
    plt.savefig(file_name)
    plt.close()

def plot_results(solutions: list, file_name: str):
    x, y = zip(*solutions)

    plt.figure(figsize=(12, 7))
    plt.scatter(x, y, color='r')
    plt.title("Best solutions")
    plt.xlabel("duration")
    plt.ylabel("cost")
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()

def cleanup(directory_path: str):
    try:
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                os.remove(file_path)
        else:
            print(f"The provided path '{directory_path}' is not a directory or does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def decision_activity_init(path: dict[str: str]):
    map_decision_activity = []
    with open(path) as file:
        data = json.load(file)
        for elem in data['resource_table']:
            role = elem['role']
            if isinstance(role, list):  # list assert
                map_decision_activity.append(role)
    return map_decision_activity

if __name__ == "__main__":

    diagram_name = "diagram_2"

    paths = {
        "jsonl": f"./output/output_{diagram_name}/communication.jsonl",
        "progression": f"./output/output_{diagram_name}/progession.png",
        "results": f"./output/output_{diagram_name}/results.png",
        "output_folder_name": diagram_name,
        "output_folder": f"./output/output_{diagram_name}",
        "petrinet_file": f"./{diagram_name}/{diagram_name}.pnml",
        "simulation_params": f"./{diagram_name}/{diagram_name}.json"
    }

    cleanup(paths["output_folder"])

    map_decision_activity = decision_activity_init(paths["simulation_params"])

    num_traces = 30
    mutation_treshold = 0.1
    n_simulations = 4

    problem = ResourceAssignmentProblem(
        map_decision_activity, 
        num_traces, 
        paths, 
        mutation_treshold, 
        n_simulations
    )

    algorithm = NSGA2(
        pop_size=20,
        sampling=IntegerRandomSampling(),
        crossover=CustomCrossover(),
        mutation=CustomMutation(),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 50)

    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=True,
        save_history=True
    )

    print(f"Best solutions found for {diagram_name}:")
    print("X:", res.X)
    print("F:", res.F)

    plot_history(res, paths["progression"])
    plot_results(res.F, paths["results"])
    