import os
from Bpmn import Bpmn
import pm4py                                                            # type: ignore
from pm4py.objects.bpmn.importer import importer as bpmn_importer       # type: ignore
from pm4py.objects.conversion.bpmn import converter as bpmn_converter   # type: ignore
import xml.etree.ElementTree as ET

class PetriNet:

    def __init__(self, bpmn_path: str):
        if os.path.exists(bpmn_path):
            bpmn_model = bpmn_importer.apply(bpmn_path)
            self.petri_net = bpmn_converter.apply(bpmn_model)
        else:
            raise ValueError(f"{bpmn_path} doesn't exist")

    def save_net(self, output_path: str):
        pm4py.write_pnml(*self.petri_net, output_path)


