
import os
import json
from pydantic import BaseModel,model_validator, Field # type: ignore

def to_lowercase(obj):
    if isinstance(obj, dict):
        return {k.lower(): to_lowercase(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_lowercase(item) for item in obj]
    elif isinstance(obj, str):
        return obj.lower()
    else:
        return obj
    
class Parameters(BaseModel):

    data: dict = Field(default_factory=dict)
    upper_bound: list[int] = Field(default_factory=list)

    input_path: str
    output_path: str
    xor_mapping: dict

    @model_validator(mode='after')
    def init(self) -> 'Parameters':

        if os.path.exists(self.input_path):
            with open(self.input_path) as file:
                self.data = json.load(file)
                self.data['tasks'] = to_lowercase(self.data.get('tasks', {}))
                self.data['roles'] = to_lowercase(self.data.get('roles', {}))
                prob: dict = self.data.get('probability', {})
                self.data['probability'] = {k.lower(): v for k, v in prob.items()}
        else:
            raise ValueError(F"{self.input_path} doesn't exists")

        self.add_mapping()
        self.compute_upper_bound()
        self.save()

        return self
    
    def add_mapping(self):
        self.data.setdefault('mapping', {})
        self.data.setdefault('probability', {})

        i = 0
        for xor, choice in self.data.get("xors", {}).items():
            if choice == "GENETICA":
                for task in self.xor_mapping[xor]:
                    self.data['mapping'][task] = i
                i += 1
            for task in self.xor_mapping[xor]:
                self.data['probability'][task] = choice

    def compute_upper_bound(self):
        for xor, choice in self.data.get("xors", {}).items():
            if choice == "GENETICA":
                self.upper_bound.append(len(self.xor_mapping[xor]) -1)
                # print(xor, self.upper_bound[-1], self.xor_mapping[xor])

    def save(self):
        with open(self.output_path, "w") as file:
            json.dump(self.data, file, indent=4)
    
    def get_upper_bound(self) -> list[int]:
        # print(self.upper_bound)
        return self.upper_bound