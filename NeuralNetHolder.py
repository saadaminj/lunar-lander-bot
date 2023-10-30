
import numpy as np
import pandas as pd
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self._layer_weights = np.loadtxt('layer_weights2.txt', dtype=float)
        self._weights = [[],[]]
        self._weights[0] = np.loadtxt('weights2.txt', dtype=float)
        self._weights[1] = np.loadtxt('weights_12.txt', dtype=float)
        self.bias = [1,1]
        
    def sigmoid (self,z):
        return 1/(1 + np.exp(-0.6*z))

    def forward_pass(self, _inputs):
        # forward propagation which multiplies weights and calculate the new layer weights after applying 
        # the activation function, sigmoid in our case
        self._layer_weights = self.sigmoid(np.dot(np.transpose(self._weights[0]), np.hstack((_inputs, [self.bias[0]]))))
        # self._layer_weights = self.sigmoid(np.dot(np.transpose(self._weights[0]), _inputs))
        # result is the resultant 2 values after our forward pass is completed with the updated weights
        _result = self.sigmoid(np.dot(np.transpose(self._weights[1]), np.hstack((self._layer_weights,[self.bias[1]]))))
        # _result = self.sigmoid(np.dot(np.transpose(self._weights[1]),self._layer_weights))
        self._layer_weights = np.transpose(np.array(self._layer_weights))
        return _result
    
    def predict(self, input_row):
        
        input = [float(x) for x in input_row.split(',')]      
        
        xmax_0, xmax_1, xmin_0, xmin_1  = (795.2104474142598, 443.3318089433691, -540.428193201501, 65.21097854954758)
        ymax_0, ymin_0, ymax_1, ymin_1  = (7.999999999999988, -2.742634266708001, 5.01526751404047, -5.603767509880821)
        
        input[0] = (min(input[0],xmax_0) - xmin_0) / (xmax_0 - xmin_0)
        input[1] = (min(input[1],xmax_1) - xmin_1) / (xmax_1 - xmin_1)
        
        output = np.transpose(self.forward_pass(np.transpose(np.array(input))))
        
        # output[0] = output[0] * (ymax_0 - ymin_0) + ymin_0
        # output[1] = output[1] * (ymax_1 - ymin_1) + ymin_1
        
        # output = [output[1] ,output[0]]
         
        print(output[0])
        output[0] =  0.7 if output[0]>0.5374 else -0.7
        return output

    

