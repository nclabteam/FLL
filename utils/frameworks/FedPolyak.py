from utils import (
    zero_parameters, 
    add_parameters, 
    divide_constant
)

class Polyak:
    def __init__(self, mix, model):
        self.mix = mix
        self.weight_sum = zero_parameters(model)
        self.num_weights = 0
    
    def add_weight(self, model):
        self.num_weights += 1
        self.weight_sum = add_parameters(
            self.weight_sum, 
            model
        )
    
    def polyak_ruppert_averaging(self):
        return divide_constant(self.weight_sum, self.num_weights)
    
    def models_mixture(self, old, new):
        for new_param, old_param, avg in zip(new.parameters(), old.parameters(), self.polyak_ruppert_averaging().parameters()):
            old_param.data = new_param.data.clone() - self.mix * (new_param.data.clone() - avg.data.clone())  
        return old