import json
import os
from typing import List, Dict, Any, Union

class Role_profile:

    def __init__(self, name: str, resources: List[str], attributes: Dict[str, Any], change_patterns: Dict[str,List[List[str]]]):
        self.name = name  
        """role of these resources"""
        self.resources = resources
        """list of resources having this role"""
        self.attributes = attributes
        """[attribute, value]"""
        self.change_patterns = change_patterns
        """[task, change_patterns (may be more than one)]"""

    def get_name(self) -> str:
        return self.name
    
    def get_resources(self) -> List[str]:
        return self.resources
    
    def get_attribute(self, attr: str) -> Union[Any, None]:
        return self.attributes.get(attr)
    
    def get_change_patterns(self, task: str) -> Union[List[List[str]], None]:
        return self.change_patterns.get(task)


class Roles(dict):

    def __init__(self, input_path: str):
        self.input_path = input_path
        """where the json is saved"""
        super().__init__()
        self.update(self.get_resources_from_file())

    def get_resources_from_file(self) -> Dict[str, Role_profile]:
        tmp_dict = {}
        if os.path.exists(self.input_path):
            with open(self.input_path) as file:
                data = json.load(file)
                roles = data.get('roles', {})
                for role_name, role_data in roles.items():
                    role_name = str(role_name).lower()
                    
                    change_patterns = role_data.get('change_patterns', {})
                    change_patterns = {k.lower(): [[str(s).lower() for s in sublist] for sublist in v] for k, v in change_patterns.items()}

                    # print(role_data)
                    tmp_dict[role_name] = Role_profile(
                        name=role_name,
                        resources=role_data.get('resources', []),
                        attributes=role_data.get('attributes', {}),
                        change_patterns=change_patterns
                    )
        else:
            raise ValueError(f"{self.input_path} doesn't exists")
        return tmp_dict
