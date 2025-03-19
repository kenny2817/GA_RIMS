import json
import os

from Bpmn import Bpmn
from Tasks import Tasks


def to_lowercase(obj):
    if isinstance(obj, dict):
        return {k.lower(): to_lowercase(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_lowercase(item) for item in obj]
    elif isinstance(obj, str):
        return obj.lower()
    else:
        return obj

class Parameters:

    def __init__(self, path: str):
        if os.path.exists(path):
            with open(path) as file:
                self.data = json.load(file)
                self.data['tasks'] = to_lowercase(self.data.get('tasks', {}))
                self.data['roles'] = to_lowercase(self.data.get('roles', {}))
                self.data['probability'] = {k.lower(): v for k, v in self.data.get('probability', {})}
        else:
            raise ValueError(F"{path} doesn't exists")
        
    
        
    def add_tasks(self, tasks: Tasks):
        for task in tasks.values():
            task_name = task.get_name()
            task_attributes = task.get_attributes()
            for role in task.get_roles():
                self.data['tasks'][f"{task_name}_{role}"] = {
                    "roles": role,
                    "attributes": task_attributes
                }

    def add_mapping(self, bpmn: Bpmn):
        xor_mapping = bpmn.get_xor_mapping()

        self.data.setdefault('mapping', {})
        self.data.setdefault('probability', {})

        i = 0
        for xor, choice in self.data.get("xors", {}).items():
            if choice == "GENETICA":
                for task in xor_mapping[xor]:
                    self.data['mapping'][task] = i
                i += 1
            for task in xor_mapping[xor]:
                self.data['probability'][task] = choice
            
        # genetica_xors = self.data.get("genetica_xors", [])
        # for i, xor in enumerate(genetica_xors):
        #     for task in xor_mapping[xor]:
        #             self.data['mapping'][task] = i
        #             self.data['probability'][task] = "GENETICA"

        # auto_xors = self.data.get("auto_xors", [])
        # for xor in auto_xors:
        #     for task in xor_mapping[xor]:
        #         self.data['probability'][task] = "AUTO"

    def save(self, path: str):
        with open(path, "w") as file:
            json.dump(self.data, file, indent=4)