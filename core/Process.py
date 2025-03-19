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

    def debug(self):
        print("\nRoles: ", [role.get_name() for role in self.get_roles().values()])

        print("\nTasks:")
        for task in self.get_tasks().values():
            print(f"{task.get_name()} can be done by {task.get_roles()}")

        print("\nCore tasks:", [task.get_name() for task in self.get_tasks().values() if task.get_iscore()])
        
        print("\nDecision trees:\n")
        for task in self.tasks.values():
            print(f"{task.get_name()} got {task.get_leaves()} leaves")
            task.print_decision_tree()
