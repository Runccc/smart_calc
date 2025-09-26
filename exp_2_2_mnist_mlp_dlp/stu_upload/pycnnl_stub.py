# Stub implementation of pycnnl for testing when actual library is not available
# This is just for testing the structure of our implementation

class IntVector:
    def __init__(self, size):
        self.data = [0] * size
        
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __getitem__(self, index):
        return self.data[index]

class CnnlNet:
    def __init__(self):
        self.input_shape = None
        self.layers = []
        self.parameters = {}
        self.input_data = None
        self.output_data = None
        
    def setInputShape(self, batch_size, input_size, h, w):
        self.input_shape = (batch_size, input_size, h, w)
        print(f"Set input shape: {self.input_shape}")
    
    def createMlpLayer(self, name, input_shape, weight_shape, output_shape):
        layer_info = {
            'name': name,
            'input_shape': input_shape.data,
            'weight_shape': weight_shape.data,
            'output_shape': output_shape.data
        }
        self.layers.append(layer_info)
        print(f"Created MLP layer {name}: input {input_shape.data} -> output {output_shape.data}")
    
    def loadParams(self, layer_idx, weights, bias):
        self.parameters[layer_idx] = {'weights': weights, 'bias': bias}
        print(f"Loaded parameters for layer {layer_idx}: weights shape {weights.shape}, bias shape {bias.shape}")
    
    def setInputData(self, data):
        self.input_data = data
        print(f"Set input data with {len(data)} elements")
    
    def forward(self):
        print("Running forward pass (stub implementation)")
        # Return dummy output for testing
        if self.input_shape:
            batch_size = self.input_shape[0]
            output_size = 10  # number of classes
            self.output_data = [0.1] * (batch_size * output_size)
        return True
    
    def getOutputData(self):
        return self.output_data if self.output_data else []