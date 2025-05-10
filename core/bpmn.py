
import os
import xml.etree.ElementTree as ET

from pydantic import BaseModel,model_validator, Field, ConfigDict # type: ignore

class Bpmn(BaseModel):

    namespace: dict[str,str] = Field(default_factory=dict)
    xor_tasks: dict = Field(default_factory=dict)
    bpmn: ET.ElementTree = Field(default=None)
    root: ET.Element = Field(default=None)
    process: ET.Element = Field(default=None)
    bpmn_tasks: list[ET.Element] = Field(default_factory=list)
    task_names: list[str] = Field(default_factory=list)

    input_path: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def init(self) -> 'Bpmn':

        if not os.path.exists(self.input_path):
            raise ValueError(f"{self.input_path} doesn't exists")
        
        self.bpmn = ET.parse(self.input_path)

        self.root = self.bpmn.getroot()
        
        self.namespace["bpmn"] = self.root.attrib.get("xmlns:bpmn", "http://www.omg.org/spec/BPMN/20100524/MODEL")

        self.process = self.root.find("bpmn:process", self.namespace)
        if self.process is None:
            raise ValueError("No <bpmn:process> found in the BPMN file.")

        self.bpmn_tasks = self.process.findall("./bpmn:task", self.namespace) + self.process.findall("./bpmn:userTask", self.namespace)
        self.task_names = [str(task.get('name')).lower() for task in self.bpmn_tasks]

        self.find_xor_mapping()
    
        return self

    def find_xor_mapping(self) -> None:
        ns = self.namespace

        tag_to_name = {
            "bpmn:task": lambda el: el[0].get("name", el[0].get("id")),
            "bpmn:parallelGateway": lambda el:  "sfl_" + el[1],
            "bpmn:exclusiveGateway": lambda el: "sfl_" + el[1],
        }

        for xor in self.process.findall("./bpmn:exclusiveGateway", ns):
            xor_id = xor.get("id")
            xor_name: str = xor.get("name", xor_id)

            outgoing_flows = xor.findall("bpmn:outgoing", ns)
            outgoing_flows = [out.text for out in outgoing_flows]
            
            task_names: list[str] = []
            for tag, get_name in tag_to_name.items():
                for elem in self.process.findall(f"./{tag}", ns):
                    incoming_flows = elem.findall("bpmn:incoming", ns)
                    incoming_flows = [inc.text for inc in incoming_flows]
                    for f in outgoing_flows:
                        if f in incoming_flows:
                            task_names.append(get_name((elem, f)))
            
            if task_names:
                self.xor_tasks[xor_name] = task_names

    def get_xor_mapping(self) -> dict:
        return self.xor_tasks 

