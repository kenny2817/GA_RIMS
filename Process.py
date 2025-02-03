from Roles import Roles
from Tasks import Tasks
from typing import List, Tuple

class Process(object):

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.roles = Roles(input_path)
        self.tasks = Tasks(input_path)

    def get_roles(self) -> Roles:
        return self.roles
    
    def get_tasks(self) -> Tasks:
        return self.tasks
    
    def gene_upper_bound(self) -> Tuple[List[str], List[int]]:
        return self.tasks.gene_upper_bound()

    def compute_decision_tree(self):
        for task in self.tasks.values():
            task.compute_decision_tree(self.roles, self.tasks)
