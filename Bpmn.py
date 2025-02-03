from Tasks import Tasks

import os
import xml.etree.ElementTree as ET


class Bpmn:
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        
        if not os.path.exists(self.input_path):
            raise ValueError(f"{self.input_path} doesn't exists")
        
        self.bpmn = ET.parse(self.input_path)

        self.process = self.bpmn.getroot()
        
        namespace = self.process.attrib.get("xmlns:bpmn", "http://www.omg.org/spec/BPMN/20100524/MODEL")
        self.namespace = {"bpmn": namespace}

        self.bpmn_tasks = self.process.findall(".//bpmn:task", self.namespace) + self.process.findall(".//bpmn:userTask", self.namespace)

    def set_core_tasks(self, tasks: Tasks):
        """
        Given a refined pst (R-PST) we can find the core tasks (all tasks in the graph)
        """
        self.task_names = [str(task.get('name')).lower() for task in self.bpmn_tasks]

        for task_name in self.task_names:
            tasks[task_name].set_iscore()

    def add_branch_and_join(self):
        """
        Inserts a branching gateway before each activity and a join gateway after.
        """
        for task in self.bpmn_tasks:
            task_id = task.get("id")

            branch_gateway = ET.Element("bpmn:parallelGateway", id=f"branch_{task_id}")
            self.process.append(branch_gateway)

            join_gateway = ET.Element("bpmn:parallelGateway", id=f"join_{task_id}")
            self.process.append(join_gateway)

            incoming_flows = task.findall("bpmn:incoming", self.namespace)
            for incoming in incoming_flows:
                incoming.text = f"flow_to_branch_{task_id}"

            outgoing_flows = task.findall("bpmn:outgoing", self.namespace)
            for outgoing in outgoing_flows:
                outgoing.text = f"flow_from_join_{task_id}"

            new_incoming_flow = ET.Element("bpmn:sequenceFlow", id=f"flow_to_task_{task_id}", sourceRef=f"branch_{task_id}", targetRef=task_id)
            self.process.append(new_incoming_flow)

            new_outgoing_flow = ET.Element("bpmn:sequenceFlow", id=f"flow_from_task_{task_id}", sourceRef=task_id, targetRef=f"join_{task_id}")
            self.process.append(new_outgoing_flow)

    def add_flow(self, flow_id: str, from_id: str, to_id: str):
        """
        Adds a new sequence flow between two elements.
        """
        new_flow = ET.Element("bpmn:sequenceFlow", id=flow_id, sourceRef=from_id, targetRef=to_id)
        self.process.append(new_flow)

    def save(self, output_path: str):
        """
        Saves the modified BPMN XML to a file.
        """
        self.bpmn.write(output_path, encoding="utf-8", xml_declaration=True)

