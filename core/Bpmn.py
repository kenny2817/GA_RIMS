import re
from xml.dom import minidom
from Tasks import Tasks

import os
import xml.etree.ElementTree as ET
import pm4py  # type: ignore
from collections import defaultdict
class Bpmn:
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        
        if not os.path.exists(self.input_path):
            raise ValueError(f"{self.input_path} doesn't exists")
        
        self.bpmn = ET.parse(self.input_path)

        self.root = self.bpmn.getroot()
        
        self.namespace = {}
        self.namespace["bpmn"] = self.root.attrib.get("xmlns:bpmn", "http://www.omg.org/spec/BPMN/20100524/MODEL")

        self.process = self.root.find("bpmn:process", self.namespace)
        if self.process is None:
            raise ValueError("No <bpmn:process> found in the BPMN file.")

        self.bpmn_tasks = self.process.findall("./bpmn:task", self.namespace) + self.process.findall("./bpmn:userTask", self.namespace)
        self.task_names = [str(task.get('name')).lower() for task in self.bpmn_tasks]

        self.find_xor_mapping()

    def get_bpmn(self):
        return self.bpmn
    def get_xor_mapping(self):
        return self.xor_tasks
    def get_tasks(self):
        return self.bpmn_tasks
    def get_task_names(self):
        return self.task_names

    def set_core_tasks(self, tasks: Tasks):
        """
        Given a refined pst (R-PST) we can find the core tasks (all tasks in the graph)
        """
        for task_name in self.task_names:
            tasks[task_name].set_iscore()
    def find_xor_mapping(self):

        if hasattr(self, 'xor_tasks'): return

        xor_tasks = {}

        ns = self.namespace

        for xor in self.process.findall("./bpmn:exclusiveGateway", ns):
            xor_id = xor.get("id")
            xor_name = xor.get("name", xor_id)

            outgoing_flows = [out.text for out in xor.findall("bpmn:outgoing", ns)]
            
            task_names = []
            tag_to_name = {
                "bpmn:task": lambda el: el.get("name", el.get("id")),
                "bpmn:parallelGateway": lambda el: "sfl_" + el.find("bpmn:incoming", ns).text,
                "bpmn:exclusiveGateway": lambda el: "sfl_" + el.find("bpmn:incoming", ns).text,
            }

            task_names = []
            for tag, get_name in tag_to_name.items():
                for elem in self.process.findall(f"./{tag}", ns):
                    incoming_flows = [inc.text for inc in elem.findall("bpmn:incoming", ns)]
                    if any(flow in incoming_flows for flow in outgoing_flows):
                        task_names.append(get_name(elem))

            
            if task_names:
                xor_tasks[xor_name] = task_names

        self.xor_tasks = xor_tasks
        # print(xor_tasks)

    def get_upper_bound(self):
        upper_bound = []
        for outgoing_task in self.xor_tasks.values():
            if len(outgoing_task) > 1:
                upper_bound.append(len(outgoing_task) -1)
        return upper_bound
        # return [len(outgoing_task) for outgoing_task in self.xor_tasks.values() if len(outgoing_task) > 1]

    def add_branch_and_join(self, tasks: Tasks):
        """
        Inserts an exclusive gateway before each task and a join gateway after,
        ensuring the tasks are properly connected to the gateways.
        """

        # elements = defaultdict(list)

        for task in self.bpmn_tasks:
            task_id = task.get("id")
            task_name = task.get("name")
            task_saved = tasks.get(task_name)
            task_tree = task_saved.get_decision_tree()

            if task_saved.get_leaves() > 1:
                # Create exclusive gateways
                branch_gateway_id = f"branch_{task_id}"
                join_gateway_id = f"join_{task_id}"
                branch_gateway = ET.SubElement(self.process, "bpmn:exclusiveGateway", id=branch_gateway_id)
                join_gateway = ET.SubElement(self.process, "bpmn:exclusiveGateway", id=join_gateway_id)


                # Redirect all incoming flows to the branch gateway
                incoming_flows = self.process.findall(f".//bpmn:sequenceFlow[@targetRef='{task_id}']", self.namespace)
                for flow in incoming_flows:
                    flow.set("targetRef", branch_gateway_id)
                    incoming_flow = ET.SubElement(branch_gateway, "bpmn:incoming")
                    incoming_flow.text = flow.get('id')

                # Create a new flow from the branch gateway to the task
                new_flow_to_task = ET.SubElement(
                    self.process,
                    "bpmn:sequenceFlow",
                    id=f"flow_from_branch_{task_id}_to_task",
                    sourceRef=branch_gateway_id,
                    targetRef=task_id,
                )
                branch_task_flow = ET.SubElement(branch_gateway, "bpmn:outgoing")
                branch_task_flow.text = new_flow_to_task.get('id')

                # Redirect all outgoing flows from the task to originate from the join gateway
                outgoing_flows = self.process.findall(f".//bpmn:sequenceFlow[@sourceRef='{task_id}']", self.namespace)
                for flow in outgoing_flows:
                    flow.set("sourceRef", join_gateway_id)
                    outgoing_flow = ET.SubElement(join_gateway, "bpmn:outgoing")
                    outgoing_flow.text = flow.get('id')

                # Create a new flow from the task to the join gateway
                new_flow_to_join = ET.SubElement(
                    self.process,
                    "bpmn:sequenceFlow",
                    id=f"flow_from_task_{task_id}_to_join",
                    sourceRef=task_id,
                    targetRef=join_gateway_id,
                )
                task_join_flow = ET.SubElement(join_gateway, 'bpmn:incoming')
                task_join_flow.text = new_flow_to_join.get('id')

                # aggiungo nuovi rami
                for r in task_tree:
                    new_task = ET.SubElement(self.process, "bpmn:task", id=f"", name=task_name)
                    new_flow = ET.SubElement(self.process, )
    def add_flow(self, flow_id: str, from_id: str, to_id: str):
        """
        Adds a new sequence flow between two elements.
        """
        new_flow = ET.Element("bpmn:sequenceFlow", id=flow_id, sourceRef=from_id, targetRef=to_id)
    def augmented_bpmn_init(self, tasks: Tasks):
        namespaces = {
            'bpmn': "http://www.omg.org/spec/BPMN/20100524/MODEL",
            'bpmndi': "http://www.omg.org/spec/BPMN/20100524/DI",
            'dc': "http://www.omg.org/spec/DD/20100524/DC",
            'di': "http://www.omg.org/spec/DD/20100524/DI"
        }

        definitions = ET.Element("bpmn:definitions", {
            "xmlns:bpmn": namespaces["bpmn"],
            "xmlns:bpmndi": namespaces["bpmndi"],
            "xmlns:dc": namespaces["dc"],
            "xmlns:di": namespaces["di"],
            "targetNamespace": "http://example.com/bpmn"
        })

        process = ET.SubElement(definitions, "bpmn:process", {
            "id": "Process_1",
            "isExecutable": "true"
        })

        tree = ET.ElementTree(definitions)
        tree.write("bpmn_diagram.xml", encoding="utf-8", xml_declaration=True)

