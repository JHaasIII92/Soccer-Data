"""
<Mapogo> (c) by <Enrique Boswell Nueve IV>

<Mapogo> is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances
from math import exp, log


class RAU(object):
    def __init__(self, nodes, Architecture, lr, key_LF, key_Opt,break_pt,**kwargs):
        self.dropout = [False,0]
        self.break_pt = break_pt

        for key, value in kwargs.items():
            if key == "dropout":
                self.dropout[0], self.dropout[1] = True, value

        self.key_LF = key_LF

        self.nodes = nodes
        self.previous_outputs = []
        self.final_output = 0
        self.all_outputs = []

        # Declare gates and stepdown network
        self.Reset_Gate = RAU_Gate(Architecture[0],lr, key_Opt)
        self.Reset_Network_Outputs = []
        self.Reset_State_Network_Outputs = []
        self.Reset_Gate_Outputs = []

        self.Update_Gate = RAU_Gate(Architecture[1],lr, key_Opt)
        self.Update_Network_Outputs = []
        self.Update_State_Network_Outputs = []
        self.Update_Gate_Outputs = []

        self.Output_Gate = RAU_Gate(Architecture[2],lr, key_Opt)
        self.Output_Network_Outputs = []
        self.Output_State_Network_Outputs = []
        self.Output_Gate_Outputs = []

        self.Attention_Gate = RAU_Network(Architecture[3],lr, key_Opt)
        self.Attention_Gate_Outputs = []

        self.Stepdown_Network = RAU_Network(Architecture[4],lr, key_Opt)
        self.Stepdown_Outputs = []

        # Parameters
        self.lr = lr  # learning rate
        self.key_Opt = key_Opt  # key for optimization

        # Error tracking
        self.error_states = []
        self.x = 0

        # BEGIN: "Loss Function"
        def Huber(targets, y, deriv=False):
            beta = 1
            if deriv:
                return ((y - targets) * (1 + ((y - targets) / beta) ** 2) ** (-0.5))
            return ((beta ** 2) * (((1 + ((y - targets) / beta) ** 2) ** 0.5) - 1)).mean()

        def MSE(targets,y,deriv=False):#calculate mean square error
            if deriv:
                return y-targets
            return np.square(targets-y).mean()

        def MAE(targets, y, deriv=False):
            if deriv:
                return (np.piecewise(y,[y < targets, y > targets, y == targets],[lambda x: -1, lambda x: 1, lambda x: 0])) * (1 /np.ma.size(y, axis=1))
            return np.sum(np.absolute(targets - y))

        def Softmax_CrossEntropy(targets, z, deriv = False, epsilon=1e-12):
            if deriv == True:
                if z.ndim == 1: z = np.reshape(z,(np.ma.size(z,axis=0),1))
                zh = z - np.amax(z,axis=0) #stablizer
                sm = (np.exp(zh) / np.sum(np.exp(zh), axis=0))
                if sm.ndim == 1: sm = np.reshape(sm,(np.ma.size(sm,axis=0),1))
                return  sm-targets

            if z.ndim == 1: z = np.reshape(z,(np.ma.size(z,axis=0),1))
            zh = z - np.amax(z,axis=0) #stablizer
            sm = (np.exp(zh) / np.sum(np.exp(zh), axis=0))
            if sm.ndim == 1: sm = np.reshape(sm,(np.ma.size(sm,axis=0),1))
            sm = np.clip(sm, epsilon, 1. - epsilon)
            sm = -1*np.sum(targets*np.log(sm+1e-3),axis=1)
            return sm

        Loss_Selector = {"MSE": MSE, "Huber": Huber, "MAE": MAE,"Softmax_CrossEntropy":Softmax_CrossEntropy,"SoftDTW":SoftDTW(),}
        self.loss_function = Loss_Selector.get(key_LF)
        # END: "Loss Function"

        def sigmoid(x, deriv=False):
            return ((1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))if deriv else (1 / (1 + np.exp(-x))))
        self.sigmoid = sigmoid

        def tanh(x, deriv=False):
            return (1 - np.tanh(x) ** 2) if deriv else np.tanh(x)
        self.tanh = tanh

        def softmax(z):
            if z.ndim == 1: z = np.reshape(z,(np.ma.size(z,axis=0),1))
            zh = z - np.amax(z,axis=0) #stablizer
            sm = (np.exp(zh) / np.sum(np.exp(zh), axis=0))
            if sm.ndim == 1: sm = np.reshape(sm,(np.ma.size(sm,axis=0),1))
            return sm
        self.softmax = softmax

        def NEAF(x, deriv=False):
            def noise(x):
                return np.abs(np.random.randn(np.size(x)))

            def linear(x, deriv=False):
                if deriv:
                    return 0.25
                return 0.25*x + 0.5

            def hard_sigmoid(x, deriv=False):
                if deriv:
                    x = np.piecewise(x,[x >= 2, (-2<x) & (x<2) ,x <= -2],[lambda x: 0, lambda x: .25 ,lambda x: 0])
                    return x
                return np.maximum(0, np.minimum(1, (x + 2) / 4))

            if deriv:
                x = linear(x)
                x = np.piecewise(x,[x > .975, (-.025<=x) & (x<=.975) ,x < .025],
                [lambda x: -.01*noise(x)*(1-hard_sigmoid(x,deriv=True))*np.exp(x-hard_sigmoid(x)),
                lambda x: 1 ,
                lambda x: .01*noise(x)*(hard_sigmoid(x,deriv=True)-1)*np.exp(hard_sigmoid(x)-x)])
                x = x*.25
                return x

            x = linear(x)
            x = np.piecewise(x ,[x < 0.025, x > 0.025],[lambda x: 0.01 * np.exp(hard_sigmoid(x) - x) * noise(x), lambda x: x])
            x = np.piecewise(x ,[x > 0.975, x < 0.975],[lambda x: 1 - 0.01 * np.exp(x-hard_sigmoid(x)) * noise(x), lambda x: x])
            return x

        self.NEAF = NEAF

    def RAU_Clear(self):
        self.previous_outputs = []
        self.Reset_Network_Outputs = []
        self.Reset_State_Network_Outputs = []
        self.Reset_Gate_Outputs = []
        self.Update_Network_Outputs = []
        self.Update_State_Network_Outputs = []
        self.Update_Gate_Outputs = []
        self.Output_Network_Outputs = []
        self.Output_State_Network_Outputs = []
        self.Output_Gate_Outputs = []
        self.Attention_Gate_Outputs = []
        self.Stepdown_Outputs = []

    def query(self,inputs):
        self.RAU_Clear()
        self.all_outputs = []
        self.State_Error, self.Past_State_Error = 0, 0
        errors = []

        for i in range(len(inputs)):
            self.previous_output = (self.previous_outputs[i-1] if i > 0 else inputs[i] * 0)

            # Update Gate Foward
            Update_Network_Output, Update_State_Network_Output = self.Update_Gate.RAU_Gate_Foward(inputs[i], self.previous_output)
            self.Update_Network_Outputs.append(Update_Network_Output)
            self.Update_State_Network_Outputs.append(Update_State_Network_Output)
            self.Update_Gate_Output = self.NEAF(Update_Network_Output + Update_State_Network_Output)
            self.Update_Gate_Outputs.append(self.Update_Gate_Output)

            # Reset Gate Foward
            Reset_Network_Output, Reset_State_Network_Output = self.Reset_Gate.RAU_Gate_Foward(inputs[i], self.previous_output)
            self.Reset_Network_Outputs.append(Reset_Network_Output)
            self.Reset_State_Network_Outputs.append(Reset_State_Network_Output)
            self.Reset_Gate_Output = self.NEAF(Reset_Network_Output + Reset_State_Network_Output)
            self.Reset_Gate_Outputs.append(self.Reset_Gate_Output)

            # Output Gate Foward
            Output_Network_Output, Output_State_Network_Output = self.Output_Gate.RAU_Gate_Foward(inputs[i], self.previous_output)
            self.Output_Network_Outputs.append(Output_Network_Output)
            self.Output_State_Network_Outputs.append(Output_State_Network_Output)
            self.Output_Gate_Output = self.tanh(Output_Network_Output+ (self.Reset_Gate_Output * Output_State_Network_Output))
            self.Output_Gate_Outputs.append(self.Output_Gate_Output)

            # Attention Gate Foward
            Attention_alpha= np.dot(self.previous_output.T,inputs[i])
            self.Attention_Gate_Output = self.Attention_Gate.RAU_Network_Foward(Attention_alpha)
            self.Attention_Gate_Outputs.append(self.Attention_Gate_Output)

            # Node output
            self.node_output = (1-self.Update_Gate_Output)*self.previous_output+(self.Update_Gate_Output*.5)*(self.Output_Gate_Output+self.Attention_Gate_Output)
            self.previous_output = self.node_output
            self.previous_outputs.append(self.previous_output)

            # Stepdown Foward
            self.final_output_stepdown = self.Stepdown_Network.RAU_Network_Foward(self.node_output)
            self.final_output = self.final_output_stepdown
            self.Stepdown_Outputs.append(self.final_output_stepdown)
            self.all_outputs.append(self.final_output_stepdown)

            if self.key_LF == "Softmax_CrossEntropy":
                self.final_output_stepdown = np.around(self.softmax(self.final_output_stepdown),decimals=4)

        return self.final_output_stepdown

    def train(self, inputs, targets,epoch):

        # BEGIN: Part One ~ Foward
        self.all_outputs = []
        self.State_Error, self.Past_State_Error = 0, 0
        errors = []
        early_break = False

        for i in range(self.nodes):
            self.previous_output = (self.previous_outputs[i-1] if i > 0 else inputs[i] * 0)

            # Update Gate Foward
            Update_Network_Output, Update_State_Network_Output = self.Update_Gate.RAU_Gate_Foward(inputs[i], self.previous_output)
            self.Update_Network_Outputs.append(Update_Network_Output)
            self.Update_State_Network_Outputs.append(Update_State_Network_Output)
            self.Update_Gate_Output = self.NEAF(Update_Network_Output + Update_State_Network_Output)
            self.Update_Gate_Outputs.append(self.Update_Gate_Output)

            # Reset Gate Foward
            Reset_Network_Output, Reset_State_Network_Output = self.Reset_Gate.RAU_Gate_Foward(inputs[i], self.previous_output)
            self.Reset_Network_Outputs.append(Reset_Network_Output)
            self.Reset_State_Network_Outputs.append(Reset_State_Network_Output)
            self.Reset_Gate_Output = self.NEAF(Reset_Network_Output + Reset_State_Network_Output)
            self.Reset_Gate_Outputs.append(self.Reset_Gate_Output)

            # Output Gate Foward
            Output_Network_Output, Output_State_Network_Output = self.Output_Gate.RAU_Gate_Foward(inputs[i], self.previous_output)
            self.Output_Network_Outputs.append(Output_Network_Output)
            self.Output_State_Network_Outputs.append(Output_State_Network_Output)
            self.Output_Gate_Output = self.tanh(Output_Network_Output+ (self.Reset_Gate_Output * Output_State_Network_Output))
            self.Output_Gate_Outputs.append(self.Output_Gate_Output)

            # Attention Gate Foward
            Attention_alpha= np.dot(self.previous_output.T,inputs[i])
            self.Attention_Gate_Output = self.Attention_Gate.RAU_Network_Foward(Attention_alpha)
            self.Attention_Gate_Outputs.append(self.Attention_Gate_Output)

            # Node output
            self.node_output = (1-self.Update_Gate_Output)*self.previous_output+(self.Update_Gate_Output*.5)*(self.Output_Gate_Output+self.Attention_Gate_Output)
            self.previous_output = self.node_output
            self.previous_outputs.append(self.previous_output)

            # Stepdown Foward
            self.final_output_stepdown = self.Stepdown_Network.RAU_Network_Foward(self.node_output)
            self.final_output = self.final_output_stepdown
            self.Stepdown_Outputs.append(self.final_output_stepdown)
            self.all_outputs.append(self.final_output_stepdown)


            # Calculate error
            error = self.loss_function(targets[i], self.final_output_stepdown)
            errors.append(error)

        if epoch%10 == 0:
            predict = np.around(self.softmax(self.final_output_stepdown),decimals=4)
            Percent_Error = np.around(np.mean(np.abs(predict-targets[-1])*100),decimals=4)
        predict = np.around(self.softmax(self.final_output_stepdown),decimals=4)
        Percent_Error = np.around(np.mean(np.abs(predict-targets[-1])*100),decimals=4)
        if Percent_Error < self.break_pt: early_break = True

        self.Error_Checker = sum(errors) / self.nodes
        self.error_states.append(sum(errors) / self.nodes)
        # End: Part One ~ Foward

        # Begin: dropout
        if self.dropout[0] == True:
            if (1-np.random.random_sample()) < self.dropout[1]:
                #print("Drop!")
                self.Update_Gate.drop()
                self.Reset_Gate.drop()
                self.Output_Gate.drop()

        # End: dropout

        # BEGIN: Part Two ~ Back
        t = self.nodes - 1
        for i in range(self.nodes):
            # Calculate output error
            self.Stepdown_Error = self.loss_function(targets[t],self.Stepdown_Network.Network_Layers[-1].output_h[t],deriv=True)
            # Calculate Stepdown Network error
            self.State_Error = self.Stepdown_Network.RAU_Network_Backward(self.Stepdown_Error, t)


            # Combine current node error and previous node errors
            if i == 0:
                self.Past_State_Error = 0
            # combine previous node state error with current state error
            self.State_Error = self.State_Error + self.Past_State_Error


            # Calculate Mesh Errors, self.Mesh_Cost_Previous_State_Error will be added to State error at end

            # This will go to Previous State Error
            self.Past_State_IRT_State_Mesh_Error = (1 - self.Update_Gate_Outputs[t]) * self.State_Error

            # This will go to output Gate
            self.State_IRT_Output_Gate_Output_Error = (self.Update_Gate_Outputs[t]*.5) * self.State_Error

            # This will go to Attention Gate
            self.State_IRT_Attention_Gate_Output_Error = (self.Update_Gate_Outputs[t]*.5) * self.State_Error

            if i == self.nodes:
                previous_output = 0
            else:
                previous_output = self.previous_outputs[t]
            self.State_IRT_Update_Gate_Output_Error = (self.Output_Gate_Outputs[t]*.5 + self.Attention_Gate_Outputs[t]*.5 - previous_output) * self.State_Error


            # Calculate Attention Gate Error
            self.Attention_Gate_Error = self.Attention_Gate.RAU_Network_Backward(self.State_IRT_Attention_Gate_Output_Error,t)
            self.Attention_State_Layer_Error = np.dot(inputs[t],self.Attention_Gate_Error)


            # Calculate Output Gate error
            self.Output_Gate_Mesh_Error = (self.tanh(self.Output_Network_Outputs[t]+self.Output_State_Network_Outputs[t], deriv=True )* self.State_IRT_Output_Gate_Output_Error)
            self.Output_Layer_Error, self.Output_State_Layer_Error = self.Output_Gate.RAU_Gate_Backward(self.Output_Gate_Mesh_Error,
                t,self.Output_Gate_Mesh_Error * self.Reset_State_Network_Outputs[t])
            self.Output_State_Layer_Error = self.Output_State_Layer_Error*self.Reset_Gate_Outputs[t]

            # Calculate Reset Gate error
            self.Output_Gate_Error_Reset_Gate = (self.Output_Gate_Mesh_Error * self.Output_Gate.State_Network.Network_Layers[-1].output_h[t])
            self.Output_Gate_Error_Reset_Gate = (self.Output_Gate_Error_Reset_Gate * self.NEAF(self.Reset_State_Network_Outputs[t], deriv=True))
            self.Reset_Layer_Error, self.Reset_State_Layer_Error = self.Reset_Gate.RAU_Gate_Backward(self.Output_Gate_Error_Reset_Gate, t)

            # Calculate Update Gate Error
            self.Update_Gate_Mesh_Error = (self.State_IRT_Update_Gate_Output_Error * self.NEAF(self.Update_Network_Outputs[t]+self.Update_State_Network_Outputs[t], deriv=True))
            self.Update_Layer_Error, self.Update_State_Layer_Error = self.Update_Gate.RAU_Gate_Backward(self.Update_Gate_Mesh_Error, t)

            # Calculate total state error in respect to previous state
            temp_1 = self.Past_State_IRT_State_Mesh_Error + self.Output_State_Layer_Error + self.Attention_State_Layer_Error
            temp_2 = self.Reset_State_Layer_Error + self.Update_State_Layer_Error
            self.Past_State_Error = temp_1 + temp_2
            t-=1
        # END: Part Two ~ Back

        # BEGIN: Part Three ~ Update
        self.Reset_Gate.RAU_Gate_Update()
        self.Update_Gate.RAU_Gate_Update()
        self.Output_Gate.RAU_Gate_Update()
        self.Attention_Gate.RAU_Network_Update()
        self.Stepdown_Network.RAU_Network_Update()
        self.RAU_Clear()
        self.x += 1
        # END: Part Three ~ Update

        return early_break


class RAU_Gate(object):
    def __init__(self, layers,lr,key_Opt):
        self.Network = RAU_Network(layers, lr,key_Opt)
        self.State_Network = RAU_Network(layers,lr, key_Opt)

    def RAU_Gate_Foward(self, inputs, previous_output):
        self.Network_Output = self.Network.RAU_Network_Foward(inputs)
        self.State_Network_Output = self.State_Network.RAU_Network_Foward(previous_output)
        return self.Network_Output, self.State_Network_Output

    def RAU_Gate_Backward(self, error, t, dif_error=np.zeros(1)):
        if dif_error.all() == 0:
            dif_error = error
        self.Network_Error = self.Network.RAU_Network_Backward(error, t)
        self.State_Network_Error = self.State_Network.RAU_Network_Backward(dif_error, t)
        return self.Network_Error, self.State_Network_Error

    def RAU_Gate_Update(self):
        self.Network.RAU_Network_Update()
        self.State_Network.RAU_Network_Update()

    def drop(self):
        self.Network.drop()
        self.State_Network.drop()


class RAU_Network(object):
    def __init__(self, layers, lr,key_Opt):
        self.Network_Layers = []
        for i in range(len(layers)):
            if layers[i][1] != 'BatchNorm':
                self.Network_Layers.append(RAU_Layer(layers[i]))
            elif layers[i][1] == 'BatchNorm':
                self.Network_Layers.append(BatchNorm(layers[i]))

        self.x = 0
        self.key_Opt = key_Opt
        self.lr = lr

        # BEGIN: "Optimizaion technique"
        def SGD(weight, gradient, j):
            #if LA.norm(gradient).any() > 1: gradient = gradient*(1/LA.norm(gradient)) #clip gradient
            weight = weight - self.lr * gradient
            return weight

        def RMSprop(weight, gradient, j):
            decay, eps = 0.9, 1e-8
            if LA.norm(gradient).any() > 1: gradient = gradient*(1/LA.norm(gradient)) #clip gradient
            self.Network_Layers[j].VdW_RMSprop = decay * self.Network_Layers[j].VdW_RMSprop + (1 - decay) * (gradient ** 2)
            weight -= (self.lr * gradient) / (np.sqrt(self.Network_Layers[j].VdW_RMSprop) + eps)
            return weight

        def AdaGrad(weight, gradient, j):
            eps = 1e-8
            if LA.norm(gradient).any() > 1: gradient = gradient*(1/LA.norm(gradient)) #clip gradient
            self.Network_Layers[j].AdaGrad_M += gradient ** 2
            weight -= (self.lr * gradient) / (np.sqrt(self.Network_Layers[j].AdaGrad_M) + eps)
            return weight

        def Adam(weight, gradient, j):
            if LA.norm(gradient).any() > 1: gradient = gradient*(1/LA.norm(gradient)) #clip gradient
            self.Network_Layers[j].Adam_M = self.Network_Layers[j].Adam_M * 0.9 + (1 - 0.9) * gradient
            self.Network_Layers[j].Adam_V = self.Network_Layers[j].Adam_V * 0.999 + (1 - 0.999) * (gradient ** 2)
            Adam_M_corrected = self.Network_Layers[j].Adam_M
            Adam_V_corrected = self.Network_Layers[j].Adam_V
            gradient = Adam_M_corrected / (np.sqrt(Adam_V_corrected) + 10e-8)
            weight -= self.lr * gradient
            return weight

        def AMSGrad(weight,gradient,j):
            if LA.norm(gradient).any() > 1: gradient = gradient*(1/LA.norm(gradient)) #clip gradient
            self.Network_Layers[j].AMSGrad_M = self.Network_Layers[j].AMSGrad_M * 0.9 + (1 - 0.9) * gradient
            self.Network_Layers[j].AMSGrad_V = self.Network_Layers[j].AMSGrad_V * 0.999 + (1 - 0.999) * (gradient ** 2)
            self.Network_Layers[j].AMSGrad_VC = np.maximum(self.Network_Layers[j].AMSGrad_V,self.Network_Layers[j].AMSGrad_VC_P)
            self.Network_Layers[j].AMSGrad_VC_P = self.Network_Layers[j].AMSGrad_VC
            weight -= (self.lr/(np.sqrt(self.Network_Layers[j].AMSGrad_VC)+10e-8))*self.Network_Layers[j].AMSGrad_M
            return weight

        Optimization_Selector = {"SGD": SGD, "AdaGrad": AdaGrad, "Adam": Adam, "RMSprop": RMSprop, "AMSGrad":AMSGrad,}
        self.opt = Optimization_Selector.get(key_Opt)
        # END: "Optimizaion technique"

    def RAU_Network_Foward(self, inputs):
        self.inputs = inputs
        for i in range(len(self.Network_Layers)):
            self.inputs = self.Network_Layers[i].foward(self.inputs)
        return self.inputs

    def RAU_Network_Backward(self, layer_delta, t):
        k = len(self.Network_Layers) - 1
        for j in range(len(self.Network_Layers)):
            layer_delta = self.Network_Layers[k].back(layer_delta,t)
            k -= 1
        return layer_delta

    def RAU_Network_Clear(self):
        for i in range(len(self.Network_Layers)):
            self.Network_Layers[i].clear()

    def RAU_Network_Update(self):
        for i in range(len(self.Network_Layers)):
            if self.Network_Layers[i].ID == "HL":
                self.Network_Layers[i].weight = self.opt(self.Network_Layers[i].weight, self.Network_Layers[i].d_weight,i)
            elif self.Network_Layers[i].ID == "BN":
                self.Network_Layers[i].gamma = self.opt(self.Network_Layers[i].gamma, self.Network_Layers[i].d_gamma, i)
                self.Network_Layers[i].beta = self.opt(self.Network_Layers[i].beta, self.Network_Layers[i].d_beta, i)
        self.RAU_Network_Clear()
        self.x+=1

    def drop(self):
        for i in range(len(self.Network_Layers)):
            if self.Network_Layers[i].ID != "BN":
                x = np.random.randint(self.Network_Layers[i].weight.shape[0])
                y = np.random.randint(self.Network_Layers[i].weight.shape[1])
                self.Network_Layers[i].weight[x,y] = 0


class RAU_Layer(object):
    def __init__(self, layer):
        self.ID = "HL" # Hidden Layer
        self.input_node = layer[0]
        self.output_node = layer[1]
        weight_range = (6 / (self.input_node + self.output_node)) ** 0.5
        self.weight = np.random.normal(-1*weight_range, weight_range, (self.input_node, self.output_node))
        self.d_weight = 0
        self.input_h = []
        self.output_h = []

        # used for RMSprop
        self.VdW_RMSprop = 0
        # used for AdaGrad
        self.AdaGrad_M = 0
        # used for Adam
        self.Adam_M, self.Adam_V = 0, 0
        # used for AMSGrad
        self.AMSGrad_M, self.AMSGrad_V, self.AMSGrad_VC,self.AMSGrad_VC_P = 0,0,0,0

        # Begin: "activation functions"
        key_AF = layer[-1]  # the last element is the key for AF

        def sigmoid(x, deriv=False):
            return ((1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x)))) if deriv else (1 / (1 + np.exp(-x))))

        def sinusoid(x, deriv=False):
            return np.cos(x) if deriv else np.sin(x)

        def hard_sigmoid(x, deriv=False):
            return (1 / 2) if deriv else np.maximum(0, np.minimum(1, (x + 1) / 2))

        def tanh(x, deriv=False):
            return (1 - (np.tanh(x)) ** 2) if deriv else np.tanh(x)

        def swish(x, deriv=False):
            return (((np.exp(-x) * (x + 1) + 1) / ((1 + np.exp(-x)) ** 2)) if deriv else x / (1 + np.exp(-x)))

        def softplus(x, deriv=False):
            return ((1 / (1 + np.exp(-x))) if deriv else (np.log(1 + np.exp(x)) / np.log(np.exp(1))))

        def ReLU(x, deriv=False):
            return np.greater(0, x).astype(float) if deriv else (x > 0) * 1

        def softsign(x, deriv=False):
            return (1 / ((1 + np.abs(x)) ** 2)) if deriv else (x / (1 + np.abs(x)))

        def gaussian(x, deriv=False):
            return -2 * x * np.exp(-x ** 2) if deriv else np.exp(-x ** 2)

        def arctan(x, deriv=False):
            return 1 / (1 + x ** 2) if deriv else np.arctan(x)

        def sinc(x, deriv=False):
            if deriv:
                return np.piecewise(x,[x == 0, x != 0],[lambda x: 0, lambda x: (np.cos(x) / (x)) + (np.sin(x) / (x ** 2))])
            return np.piecewise(x, [x == 0, x != 0], [lambda x: 1, lambda x: np.sin(x) / (x)])

        def SineReLU(x, deriv=False):
            if deriv:
                return np.piecewise(x, [x > 0, x <= 0], [lambda x: 1, lambda x: np.cos(x) + np.sin(x)])
            return np.piecewise(x,[x > 0, x <= 0],[lambda x: x, lambda x: 10e-2 * (np.sin(x) - np.cos(x))])

        def softmax(z,deriv=False):
            if deriv == True:
                block = []
                if z.ndim == 1: z = np.reshape(z,(np.ma.size(z,axis=0),1))
                inputs = z
                z = softmax(inputs)
                def grad(x,i):
                    x = x.reshape(-1,1)
                    x = np.diagflat(x) - np.dot(x, x.T)
                    x = x.dot(inputs[:,i])
                    block.append(x)
                [grad(z[:,i],i) for i in range(np.ma.size(z,axis=1))]
                return np.array(block).T
            if z.ndim == 1: z = np.reshape(z,(np.ma.size(z,axis=0),1))
            zh = z - np.amax(z,axis=0) #stablizer
            sm = (np.exp(zh) / np.sum(np.exp(zh), axis=0))
            if sm.ndim == 1: sm = np.reshape(sm,(np.ma.size(sm,axis=0),1))
            return sm

        def NEAF(x, deriv=False):
            def noise(x):
                return np.abs(np.random.randn(np.size(x)))

            def linear(x, deriv=False):
                if deriv:
                    return 0.25
                return 0.25*x + 0.5

            def hard_sigmoid(x, deriv=False):
                if deriv:
                    x = np.piecewise(x,[x >= 2, (-2<x) & (x<2) ,x <= -2],[lambda x: 0, lambda x: .25 ,lambda x: 0])
                    return x
                return np.maximum(0, np.minimum(1, (x + 2) / 4))

            if deriv:
                x = linear(x)
                x = np.piecewise(x,[x > .975, (-.025<=x) & (x<=.975) ,x < .025],
                [lambda x: -.01*noise(x)*(1-hard_sigmoid(x,deriv=True))*np.exp(x-hard_sigmoid(x)),
                lambda x: 1 ,
                lambda x: .01*noise(x)*(hard_sigmoid(x,deriv=True)-1)*np.exp(hard_sigmoid(x)-x)])
                x = x*.25
                return x

            x = linear(x)
            x = np.piecewise(x ,[x < 0.025, x > 0.025],[lambda x: 0.01 * np.exp(hard_sigmoid(x) - x) * noise(x), lambda x: x])
            x = np.piecewise(x ,[x > 0.975, x < 0.975],[lambda x: 1 - 0.01 * np.exp(x-hard_sigmoid(x)) * noise(x), lambda x: x])
            return x

        def linear(x,deriv=False):
            if deriv: return 1
            return x

        Activation_Function_Selector = { "ReLU": ReLU, "hard_sigmoid": hard_sigmoid, "tanh": tanh, "swish": swish,
            "softplus": softplus, "sigmoid": sigmoid, "softsign": softsign, "sinusoid": sinusoid, "gaussian": gaussian,
            "sinc": sinc,"arctan": arctan, "SineReLU": SineReLU, "softmax": softmax, "NEAF":NEAF, "linear": linear,
        }
        self.activation_function = Activation_Function_Selector.get(key_AF)
        # END: "activation functions"

    def foward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation_function(np.dot(self.weight.T, self.inputs))
        self.input_h.append(self.inputs)
        self.output_h.append(self.outputs)
        return self.outputs

    def back(self, error, t):
        self.layer_delta = error * self.activation_function(np.dot(self.weight.T, self.input_h[t]), deriv=True)
        gradient = np.dot(self.input_h[t], self.layer_delta.T)
        self.d_weight += gradient
        self.layer_delta = np.dot(self.weight, self.layer_delta)
        return self.layer_delta

    def clear(self):
        self.d_weight = 0
        self.input_h = []
        self.output_h = []


class BatchNorm():
    def __init__(self,number_variables):
        self.ID = 'BN' #BatchNorm
        self.x_h = []
        self.xh_h = []
        self.y_h = []
        self.batch_mean_h = []
        self.batch_variance_h = []

        self.inputs = number_variables[0]
        self.batch_mean, self.batch_variance, self.y, self.xh = 0, 0, 0 , 0
        self.gamma = np.ones((self.inputs, 1))
        self.beta = np.ones((self.inputs,1))

        self.d_xh, self.d_variance, self.d_x, self.d_mean = 0, 0, 0, 0
        self.d_gamma, self.d_beta = 0, 0
        self.layer_delta = 0

        # Factors
        # used for RMSprop
        self.VdW_RMSprop = 0
        # used for AdaGrad
        self.AdaGrad_M = 0
        # used for Adam
        self.Adam_M, self.Adam_V = 0, 0
        # used for AdamWR
        self.AdamWR_M, self.AdamWR_V = 0, 0
        self.AdamWR_Tcur = 0
        self.AdamWR_Ti = 20  # after every 20 epochs reset
        self.AdamWR_weight_norm = 0.1  # weight decay correction
        self.AdamWR_decay_weight = self.AdamWR_weight_norm * ((1 / (4 * 100)) ** 0.5)  # ((batches per epoch/(samples per batch*total epochs))**.5
        # used for AMSGrad
        self.AMSGrad_M, self.AMSGrad_V, self.AMSGrad_VC,self.AMSGrad_VC_P = 0,0,0,0

    def foward(self,x):
        self.x = x
        self.batch_mean = np.array(np.sum(self.x, axis=0) / np.size(self.x, axis=0), ndmin=2)
        self.batch_variance = np.array(np.sum((self.x - self.batch_mean) ** 2, axis=0) / np.size(self.x, axis=0),ndmin=2)
        self.xh = (self.x-self.batch_mean) / ((self.batch_variance + 1e-8)**.5)
        self.y = (self.xh * self.gamma)+self.beta

        self.batch_variance_h.append(self.batch_variance)
        self.batch_mean_h.append(self.batch_mean)
        self.x_h.append(self.x)
        self.xh_h.append(self.xh)
        self.y_h.append(self.y)
        return self.y

    def back(self,error,t):
        self.d_xh = error * self.gamma

        self.d_variance = -.5 * np.sum(self.d_xh * (self.x_h[t] - self.batch_mean_h[t]) * ((self.batch_variance_h[t]+1e-8)**(-1.5)), axis = 0)

        self.d_mean = np.sum(self.d_xh * (-1 / ((self.batch_variance_h[t]+1e-8)**.5)),axis = 0)

        self.layer_delta = self.d_xh * (1 / ((self.batch_variance_h[t]+1e-8)**.5))
        self.layer_delta = self.layer_delta+(self.d_variance * (2/np.ma.size(self.x,axis=0)) * (self.x_h[t] - self.batch_mean_h[t]))
        self.layer_delta = self.layer_delta+(self.d_mean * (1/np.ma.size(self.x,axis=0)))
        self.layer_delta = np.array(self.layer_delta,ndmin=2)

        self.d_gamma += np.array(np.sum(error * self.xh_h[t],axis=1),ndmin=2).T
        self.d_beta += np.array(np.sum(error,axis=1),ndmin=2).T
        return self.layer_delta

    def clear(self):
        self.d_xh, self.d_variance, self.d_x, self.d_mean = 0, 0, 0, 0
        self.x_h, self.xh_h, self.y_h, self.batch_mean_h, self.batch_variance_h = [], [], [], [], []
        self.d_gamma, self.d_beta = 0, 0
        self.layer_delta = 0


class SoftDTW(object):
    def __init__(self, gamma=0.1):
        self.gamma = gamma
        self.D = None
        self.D_foward = None
        self.R_foward = None

    def softmin3(self, a, b, c, gamma):
        a /= -gamma
        b /= -gamma
        c /= -gamma
        max_val = max(max(a, b), c)

        tmp = 0
        tmp += exp(a - max_val)
        tmp += exp(b - max_val)
        tmp += exp(c - max_val)

        return -gamma * (log(tmp) + max_val)

    def SquaredEuclidean(self):
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.float32)
        #if self.X.ndim == 1:
        #    self.X, self.Y = self.X.reshape(1,-1), self.Y.reshape(1,-1)
        return euclidean_distances(self.X, self.Y, squared=True)

    def softDTW(self, D):
        self.D = D
        self.D = self.D.astype(np.float32)
        m = self.D.shape[0]
        n = self.D.shape[1]
        self.R = np.zeros((m + 2, n + 2), dtype=np.float32)

        for i in range(m + 1):
            self.R[i, 0] = 1.7e308

        for j in range(n + 1):
            self.R[0, j] = 1.7e308

        self.R[0, 0] = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # D is indexed starting from 0.
                self.R[i, j] = self.D[i - 1, j - 1] + self.softmin3(
                    self.R[i - 1, j], self.R[i - 1, j - 1], self.R[i, j - 1], self.gamma
                )

        self.D_foward = self.D
        self.R_foward = self.R

        return self.R[m, n]

    def __call__(self, X, Y,deriv=False):
        if deriv==True:
            self.X, self.Y = X, Y
            #assert self.X.shape[1] == self.Y.shape[1]
            self.D = self.SquaredEuclidean()
            self.error = self.softDTW(self.D)
            return self.backward()
        else:
            targets = X
            error = np.mean((np.abs(Y-targets)/targets))
            return error

    def soft_dtw_grad(self, D, R, E, gamma):
        m = D.shape[0] - 1
        n = D.shape[1] - 1

        for i in range(1, m + 1):
            # For D, indices start from 0 throughout.
            D[i - 1, n] = 0
            R[i, n + 1] = -1.7e308

        for j in range(1, n + 1):
            D[m, j - 1] = 0
            R[m + 1, j] = -1.7e308

        E[m + 1, n + 1] = 1
        R[m + 1, n + 1] = R[m, n]
        D[m, n] = 0

        for j in reversed(range(1, n + 1)):  # ranges from n to 1
            for i in reversed(range(1, m + 1)):  # ranges from m to 1
                a = exp((R[i + 1, j] - R[i, j] - D[i, j - 1]) / gamma)
                b = exp((R[i, j + 1] - R[i, j] - D[i - 1, j]) / gamma)
                c = exp((R[i + 1, j + 1] - R[i, j] - D[i, j]) / gamma)
                E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c

        return E

    def jacobian_product(self, E):
        G = np.zeros_like(self.X)
        G = self.jacobian_product_sq_euc(self.X, self.Y, E, G)
        return G

    def jacobian_product_sq_euc(self, X, Y, E, G):
        m = X.shape[0]
        n = Y.shape[0]
        d = X.shape[1]

        for i in range(m):
            for j in range(n):
                for k in range(d):
                    G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])
        return G

    def backward(self):
        m, n = self.D.shape
        D = np.vstack((self.D, np.zeros(n)))
        D = np.hstack((D, np.zeros((m + 1, 1))))

        E = np.zeros((m + 2, n + 2))
        E = self.soft_dtw_grad(D, self.R, E, self.gamma)
        E = E[1:-1, 1:-1]
        gZ = self.jacobian_product(E)
        return -1*gZ
