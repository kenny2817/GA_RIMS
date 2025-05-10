
import os
import pm4py  # type: ignore
from pm4py.objects.bpmn.importer import importer as bpmn_importer       # type: ignore
from pm4py.objects.conversion.bpmn import converter as bpmn_converter   # type: ignore
                                                          
from pydantic import BaseModel,model_validator, Field, ConfigDict # type: ignore

class PetriNet(BaseModel):

    petri_net: tuple = Field(default=None)

    input_path: str
    output_path: str

    @model_validator(mode='after')
    def init(self) -> 'PetriNet':
        if os.path.exists(self.input_path):
            bpmn_model = bpmn_importer.apply(self.input_path)
            self.petri_net = bpmn_converter.apply(bpmn_model)
        else:
            raise ValueError(f"{self.input_path} doesn't exist")
        
        self.save_net()
        
        return self

    def save_net(self):
        pm4py.write_pnml(*self.petri_net, self.output_path)