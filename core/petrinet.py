
import os
import pm4py  # type: ignore
from pm4py.objects.bpmn.importer import importer as bpmn_importer       # type: ignore
from pm4py.objects.conversion.bpmn import converter as bpmn_converter   # type: ignore
                                                          
from pydantic import BaseModel,model_validator, Field # type: ignore

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
        
        self.make_skip()
        
        return self
    
    def make_skip(self):
        petri = self.petri_net[0]
        trans = petri.transitions
        for t in trans:
            if 'skip' in str(t.label):
                t.label = None


if __name__ == "__main__":
    p = PetriNet(
        input_path="diagrams/ignored/test_0.bpmn",
        output_path="diagrams/ignored/test_0.pnml"
    )