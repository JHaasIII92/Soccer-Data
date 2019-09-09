"""
Author of Code: Enrique Boswell Nueve IV.
Email: enriquenueve9@gmail.com
Written: June 2019

Description: Python-3.7 implementation of the paper "A General Reflex Fuzzy Min-Max Neural Network."
The General Reflex Fuzzy Min-Max Neural Network is a supervised clustering algorithim.

I DO NOT CLAIM TO BE THE CREATOR OF THIS ALGORITHIM! I have no affilation
with the creation of the paper this code is based on.
For information about the creators of this algorithim and the paper that it is based on,
please refer to the citation below.

@article{Nandedkar2007AGR,
  title={A General Reflex Fuzzy Min-Max Neural Network},
  author={Abhijeet V. Nandedkar and Prabir Kumar Biswas},
  journal={Engineering Letters},
  year={2007},
  volume={14},
  pages={195-205}
}
"""


# --- Impoer modules --- #
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split


# --- General Reflex Fuzzy Min-Max Neural Network Class --- #
class ReflexFuzzyNeuroNetwork(object):
    def __init__(self, gamma=1, theta=0.1):
        self.gamma = gamma
        self.theta = theta
        self.classes = []

        self.HyperBoxes = []
        self.OCN_HyperBoxes = []
        self.CCN_HyperBoxes = []

        self.Containment_Check_bool = False

    def __call__(self):
        a = '"A fuzzy neural network or neuro-fuzzy system is a learning machine\n'
        b = "that finds the parameters of a fuzzy system by exploting approximation\n"
        c = 'techniques from a neural network.(Kruse, Scholarpedia)" '
        print(a+b+c)

    def __str__(self):
        return "Gamma: %s \nTheta: %s \nNumber boxes: %s" % (
            self.gamma,
            self.theta,
            len(self.HyperBoxes),
        )

    def print_boxes(self):
        [print("Box~" + str(i) + ": " + str(dict(self.HyperBoxes[i])))for i in range(len(self.HyperBoxes))]
        [print("OCN Box~" + str(i) + ": " + str(dict(self.OCN_HyperBoxes[i])))for i in range(len(self.OCN_HyperBoxes))]
        [print("CCN Box~" + str(i) + ": " + str(dict(self.CCN_HyperBoxes[i])))for i in range(len(self.CCN_HyperBoxes))]

    def query(self, pt, validation=False):

        # --- Query CLN nodes --- #
        CLN_max_values = [None] * len(self.classes)
        for i in range(len(self.classes)):
            for j in range(len(self.HyperBoxes)):
                if self.HyperBoxes[j]['class'] == i+1:
                    value = self.MembershipValue(pt, dict(self.HyperBoxes[j]))
                    if CLN_max_values[i] == None:
                        CLN_max_values[i] = value
                    elif CLN_max_values[i] < value:
                        CLN_max_values[i] = value



        # --- Query OCN nodes --- #
        OCN_min_values = [None] * len(self.classes)
        for i in range(len(self.classes)):
            for j in range(len(self.OCN_HyperBoxes)):
                if self.OCN_HyperBoxes[j]['class'] == i + 1:
                    value = self.OCN_Activation_Function(pt,dict(self.OCN_HyperBoxes[j]))
                    if OCN_min_values[i] == None:
                        OCN_min_values[i] = value
                    elif OCN_min_values[i] > value:
                        OCN_min_values[i] = value


        # --- Query CCN nodes --- #
        CCN_min_values = [None] * len(self.classes)
        for i in range(len(self.classes)):
            for j in range(len(self.CCN_HyperBoxes)):
                if self.CCN_HyperBoxes[j]['class'] == i + 1:
                    value = self.CCN_Activation_Function(pt,dict(self.CCN_HyperBoxes[j]))
                    if CCN_min_values[i] == None:
                        CCN_min_values[i] = value
                    elif CCN_min_values[i] > value:
                        CCN_min_values[i] = value


        # --- Correct CCN values if value is  None --- #
        for i in range(len(self.classes)):
            if CLN_max_values[i] == None:
                CLN_max_values[i] = 0


        # --- Correct CCN values if value is  None --- #
        for i in range(len(self.classes)):
            if CCN_min_values[i] == None or 'nan':
                CCN_min_values[i] = 2


        # --- Correct CCN values if value is  None --- #
        for i in range(len(self.classes)):
            if OCN_min_values[i] == None or 'nan':
                OCN_min_values[i] = 0


        # --- Compute Compensation Values from OCN and CCN values --- #

        Compensation_Values = []
        for i in range(len(self.classes)):
            if CCN_min_values[i] <= OCN_min_values[i]:
                Compensation_Values.append(CCN_min_values[i])
            else:
                Compensation_Values.append(OCN_min_values[i])

        return [sum(x) for x in zip(CLN_max_values, Compensation_Values)]



    def test(self, Y, labels):
        print("Needs to be made ;)")


    # --- Activation functions for Hyperboxes --- #
    def ThresholdFunction(self,x):
        if x >= 0: return 1
        else: return 0

    def MembershipValue(self, test_pt, box):
        min_pt, max_pt = box["min"], box["max"]
        n_dim = len(test_pt)
        membership_value = np.zeros(n_dim)

        def Ramp_function(x):
            x = x*self.gamma
            ramp_value = np.piecewise(x, [x < 0, 0 <= x  <= 1, x > 1], [0,x,1])
            return ramp_value

        for i in range(n_dim):
            membership_value[i] = min([1 - Ramp_function(test_pt[i]-max_pt[i]), 1 - Ramp_function(min_pt[i]-test_pt[i])])

        return min(membership_value) # membership_value is an np array

    def OCN_Activation_Function(self,test_pt,box):
        temp = self.ThresholdFunction(self.MembershipValue(test_pt,box) - 1)
        sum_value = 0
        for i in range(len(test_pt)):
            sum_value += max(test_pt[i]/box['max'][i],box['min'][i]/test_pt[i])

        return (-1 + (1/len(test_pt))*sum_value) * temp

    def CCN_Activation_Function(self,test_pt,box):
        a = (-1)*self.ThresholdFunction(self.MembershipValue(test_pt,box) - 1)
        return a


    # --- Functions to create new Hyperbox --- #
    def add_box(self, pt, label):
        new_box = OrderedDict([("min", pt), ("max", pt), ("class", label)])
        self.HyperBoxes.append(new_box)

    # --- Functions to create new OCN Hyperbox --- #
    def add_OCN_box(self,index,test_box):
        labels = sorted([self.HyperBoxes[index]["class"],self.HyperBoxes[test_box]["class"]])
        new_max, new_min = [], []
        for i in range(len(list(self.HyperBoxes[test_box]["max"]))):
            new_min.append(max(self.HyperBoxes[index]["min"][i],self.HyperBoxes[test_box]["min"][i]))
            new_max.append(min(self.HyperBoxes[index]["max"][i],self.HyperBoxes[test_box]["max"][i]))
        new_box_1 = OrderedDict([("min", new_min), ("max", new_max), ("class", labels[0]),])
        new_box_2 = OrderedDict([("min", new_min), ("max", new_max), ("class", labels[1]),])
        self.OCN_HyperBoxes.append(new_box_1)
        self.OCN_HyperBoxes.append(new_box_2)

    # --- Functions to create new CCN Hyperbox --- #
    def add_CCN_box(self,index,test_box,case):
        if case == 1:
            label = self.HyperBoxes[index]["class"]
            new_box = OrderedDict([("min", self.HyperBoxes[test_box]["min"]), ("max",self.HyperBoxes[test_box]["max"] ), ("class", label),])
            self.CCN_HyperBoxes.append(new_box)
        elif case == 2:
            label = self.HyperBoxes[test_box]["class"]
            new_box = OrderedDict([("min", self.HyperBoxes[index]["min"]), ("max",self.HyperBoxes[index]["max"] ), ("class", label),])
            self.CCN_HyperBoxes.append(new_box)


    # --- Connect two points that meet membership critera of the same class to form new Hyperbox --- #
    def Expand(self, pt, label):

        pt = OrderedDict([("min", pt), ("max", pt), ("class", label)])

        meet_criteria_value, meet_criteria_index = [], []
        expanded_index = None

        # check expansion criterion
        for j in range(len(self.HyperBoxes)):
            if self.HyperBoxes[j]["class"] == pt["class"]:
                alpha = np.sum(np.maximum(self.HyperBoxes[j]["max"], pt["max"])- np.minimum(self.HyperBoxes[j]["min"], pt["min"]))

                if len(pt["min"]) * self.theta >= alpha - 10e-7:
                    meet_criteria_value.append(alpha)
                    meet_criteria_index.append(j)
                    expanded_index = j

        # Expansion critertion not meet, thus add new box
        if not meet_criteria_value:
            self.add_box(pt["min"], label)
        else: # Expansion critertion met and adjusting box
            min_value_index = meet_criteria_value.index(min(meet_criteria_value))
            box_min_index = meet_criteria_index[min_value_index]
            new_box = OrderedDict(
                [
                    ("min",np.minimum(self.HyperBoxes[box_min_index]["min"], pt["min"]),),
                    ("max",np.maximum(self.HyperBoxes[box_min_index]["max"], pt["max"]),),
                    ("class", self.HyperBoxes[box_min_index]["class"]),
                ]
            )
            del self.HyperBoxes[box_min_index]
            self.HyperBoxes.append(new_box)
        return expanded_index


    # --- Reshape boxes that are overlapping or being contained. Also create OCN or CCN box while performing contraction --- #
    def Contraction(self, index):
        for test_box in range(len(self.HyperBoxes)):
            # --- Check if two boxes being compared are of the same class, if so, no need for contraction --- #
            if self.HyperBoxes[test_box]["class"] == self.HyperBoxes[index]["class"]:
                continue

            # --- Declare placeholders for checking overlap or containment of boxes --- #
            min_expand, max_expand = self.HyperBoxes[index]["min"], self.HyperBoxes[index]["max"]
            test_min, test_max = self.HyperBoxes[test_box]["min"], self.HyperBoxes[test_box]["max"]
            detlta_new = delta_old = 1
            min_overlap_index = 1

            # --- Containment Check --- #
            self.Containment_Check_bool == False

            # --- Case 1 of contaiment --- #
            if all(list(self.HyperBoxes[test_box]["max"] < self.HyperBoxes[index]["max"])):
                if all(list(self.HyperBoxes[index]["min"] < self.HyperBoxes[test_box]["min"])):
                    self.add_CCN_box(index,test_box,case=1)
                    self.Containment_Check_bool = True

            # --- Case 2 of contaiment --- #
            if all(list(self.HyperBoxes[index]["max"] < self.HyperBoxes[test_box]["max"])):
                if all(list(self.HyperBoxes[test_box]["min"] < self.HyperBoxes[index]["min"])):
                    self.add_CCN_box(index,test_box,case=2)
                    self.Containment_Check_bool = True

            # --- Overlap Check --- #
            for j in range(len(min_expand)):
                if min_expand[j] < test_min[j] < max_expand[j] < test_max[j]:
                    detlta_new = min(delta_old, max_expand[j] - test_min[j])
                elif test_min[j] < min_expand[j] < test_max[j] < max_expand[j]:
                    detlta_new = min(delta_old, test_max[j] - min_expand[j])
                elif min_expand[j] < test_min[j] < test_max[j] < max_expand[j]:
                    detlta_new = min(delta_old,min(max_expand[j] - test_min[j], test_max[j] - min_expand[j]),)
                elif test_min[j] < min_expand[j] < max_expand[j] < test_max[j]:
                    detlta_new = min(delta_old,max_expand[j] - test_min[j],test_max[j] - min_expand[j],)

                if delta_old - detlta_new > 0:
                    min_overlap_index = j
                    delta_old = detlta_new

            # --- Contraction --- #
            if min_overlap_index >= 0:
                i = min_overlap_index
                if min_expand[i] < test_min[i] < max_expand[i] < test_max[i]:
                    if self.Containment_Check_bool == False: self.add_OCN_box(index,test_box)
                    test_min[i] = max_expand[i] = (test_min[i] + max_expand[i]) / 2
                elif test_min[i] < min_expand[i] < test_max[i] < max_expand[i]:
                    if self.Containment_Check_bool == False: self.add_OCN_box(index,test_box)
                    min_expand[i] = test_max[i] = (min_expand[i] + test_max[i]) / 2
                elif min_expand[i] < test_min[i] < test_max[i] < max_expand[i]:
                    if (max_expand[i] - test_min[i]) > (test_max[i] - min_expand[i]):
                        if self.Containment_Check_bool == False: self.add_OCN_box(index,test_box)
                        min_expand[i] = test_max[i]
                    else:
                        if self.Containment_Check_bool == False: self.add_OCN_box(index,test_box)
                        max_expand[i] = test_min[i]
                elif test_min[i] < min_expand[i] < max_expand[i] < test_max[i]: # case 4
                    if (test_max[i] - min_expand[i]) > (max_expand[i] - test_min[i]):
                        if self.Containment_Check_bool == False: self.add_OCN_box(index,test_box)
                        test_min[i] = max_expand[i]
                    else:
                        if self.Containment_Check_bool == False: self.add_OCN_box(index,test_box)
                        test_max[i] = min_expand[i]

                # --- Reshape boxes --- #
                self.HyperBoxes[test_box]["min"], self.HyperBoxes[test_box]["max"] = test_min, test_max
                self.HyperBoxes[index]["min"], self.HyperBoxes[index]["max"] = min_expand, max_expand


    # --- Learn hyperbox parameters --- #
    def train(self, X, labels):
        # --- Convert data type to np array --- #
        X = X.astype(np.float32)
        labels = labels.astype(np.int32)
        n_samples = X.shape[1]

        # --- Check each sample has coresponding label --- #
        if n_samples != labels.shape[0]:
            raise ValueError("Number of samples, "+ str(n_samples)+ ", and number of labels, "
                + str(labels.shape[0])+ ", do not match.")

        # ---  Analyze samples --- #
        self.add_box(X[:, 0], labels[0])  # add intial box
        self.classes.append(labels[0])  # add intial label

        for i in range(1, n_samples):  # add other boxes
            if labels[i] not in self.classes:  # Check for newly introduced class
                self.add_box(X[:, i], labels[i])
                self.classes.append(labels[i])
            else:
                expanded_index = self.Expand(X[:, i], labels[i])  # Check expansion critertion
                if expanded_index != None:
                    self.Contraction(expanded_index)  # If expansion occured, possible overlap!!!


def main():
    # --- Import Iris data and split --- #
    data = pd.read_csv('iris_data_norm.csv')
    data = data.iloc[:,1:]
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    X_train, X_test = X_train.T, X_test.T

    # --- Declare network --- #
    nn = ReflexFuzzyNeuroNetwork(gamma=4, theta=.1)

    # --- Run network --- #
    nn.train(X_train, y_train)

    # --- Print formed hyperboxes --- #
    #nn.print_boxes()

    # --- Query point --- #
    print('Prediction: ',nn.query(X_test[:,9].T))
    print('Label: ', y_test[9])
