import json
import os
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple, Union
if TYPE_CHECKING:
    from Tasks import Tasks

import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from Roles import Roles

class Task_profile(object):

    def __init__(self, name: str, roles: List[str], attributes: Dict[str, Any]):
        self.name = name
        """task name"""
        self.iscore = False
        """true if task is core of the process"""
        self.roles = roles
        """roles assigned to this task"""
        self.attributes = attributes
        """[attribute, value]"""
        self.decision_tree = None
        """tree structure representing the decision tree"""
        self.leaves = 0
        """number of leaves"""
    
    def get_name(self) -> str:
        return self.name
    
    def get_iscore(self) -> bool:
        return self.iscore
    
    def get_roles(self) -> List[str]:
        return self.roles

    def get_attribute(self, attr: str) -> Union[Any, None]:
        return self.attributes.get(attr, None)
    
    def get_leaves(self) -> int:
        return self.leaves
    
    def get_decision_tree(self):
        return self.decision_tree
    
    def set_iscore(self):
        self.iscore = True
    
    def compute_decision_tree(self, roles: Roles, tasks: "Tasks"):
        if self.decision_tree is not None:
            return
        core_elem = ET.Element("Task", name=self.get_name())
        for role in self.roles:
            role_elem = ET.SubElement(core_elem, "Role", role=role)
            role_data = roles.get(role, None)

            if role_data is None:
                print(role)
                raise ValueError(f"role {role} is not defined in input file")
            
            change_patterns = role_data.get_change_patterns(self.name) or []
            if len(change_patterns) == 0:
                self.leaves += 1

            for change_pattern in change_patterns:
                # print(change_pattern)
                change_type = str(change_pattern[0]).lower()
                x = str(change_pattern[1]).lower()
                y = str(change_pattern[2]).lower()
                direction = str(change_pattern[3]).lower()
                
                if change_type in ['insert', 'replace']: # replace not clear at all
                    change_elem = ET.SubElement(role_elem,  f"{change_type}", x=x, y=y, direction=direction)
                    new_task = tasks.get(x)
                    # print(new_task.get_name(), x)
                    if new_task:
                        if new_task.get_iscore():
                            self.leaves += 1
                            ET.SubElement(change_elem, "Task", name=new_task.get_name())
                        else:
                            new_task.compute_decision_tree(roles, tasks)
                            change_elem.append(new_task.get_decision_tree())
                            self.leaves += new_task.get_leaves()

                elif change_type == 'delete':
                    change_elem = ET.SubElement(role_elem,  f"{change_type}", x=x, y=y, direction=direction)
                    self.leaves += 1

                else:
                    raise ValueError(f"{change_type} is not handled as change pattern, handled: [insert, replace, delete]")

        self.decision_tree = core_elem
        return core_elem

    def print_decision_tree(self):
        if self.decision_tree is None:
            print("Decision tree is empty!")
            return

        tree_str = ET.tostring(self.decision_tree, encoding="unicode")
        
        if isinstance(tree_str, bytes):
            tree_str = tree_str.decode("utf-8")
        
        pretty_xml = parseString(tree_str).toprettyxml(indent="  ")
        print(pretty_xml)



class Tasks(dict):

    def __init__(self, input_path: str):
        self.input_path = input_path
        """where the json is saved"""
        super().__init__()
        self.update(self.get_tasks_from_file())
    
    def get_tasks_from_file(self) -> Dict[str, Task_profile]:
        if not os.path.exists(self.input_path):
            raise ValueError(f"<{self.input_path}> doesn't exists")
        
        tmp_dict = {}
        with open(self.input_path) as file:
            data = json.load(file)
            tasks = data.get('tasks', [])
            for task_name, task_data in tasks.items():
                task_name = str(task_name).lower()
                roles = task_data.get('roles', [])
                if not isinstance(roles, list):
                    roles = [roles]
                roles = [str(r).lower() for r in roles]
                tmp_dict[task_name] = Task_profile(
                    name=task_name,
                    roles=roles,
                    attributes=task_data.get('attributes', {})
                )
        return tmp_dict


    def gene_upper_bound(self) -> Tuple[List[str], List[int]]:
        if not hasattr(self, "upper_bound_names") or not hasattr (self, "upper_bound_values"):
            self.upper_bound_names = []
            self.upper_bound_values = []

            for task_name, task_data in self.items():
                if task_data.get_iscore() and task_data.get_leaves() > 1:
                    self.upper_bound_names.append(task_name)
                    self.upper_bound_values.append(task_data.get_leaves())

        return self.upper_bound_names, self.upper_bound_values