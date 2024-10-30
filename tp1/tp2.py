# Implementation of value iteration : V = TV

import numpy as np
import matplotlib.pyplot as plt

# graph with two states 0 and 1
# define the reward matrix
# state 0 to 0
r0a1 = 1
# state 0 to 1
r0a2 = 10
# state 1 to 1
r1a1 = 0
# state 1 to 0
r1a2 = -15

# epsilon to fix
epsilon = 0.01

# T operator for V = TV
def T(a, Vn0, Vn1, gamma):
    if a == 0:
        if r0a1+gamma*Vn0 > r0a2+gamma*Vn1:
            return r0a1+gamma*Vn0, 1
        else:
            return r0a2+gamma*Vn1, 2
    elif a == 1:
        if r1a1+gamma*Vn1 > r1a2+gamma*Vn0:
            return r1a1+gamma*Vn1, 1
        else:
            return r1a2+gamma*Vn0, 2

# value iteration algorithm
def value_iteration(gamma):
    count = 0
    # we fix an arbitrary initial value for V
    Vn = np.zeros(2)
    # we initialize the value of Vn+1
    Vn1 = np.zeros(2)

    Vn1[0],a0 = T(0, Vn[0], Vn[1], gamma)
    Vn1[1],a1 = T(1, Vn[0], Vn[1], gamma)
    
    while max(abs(Vn1 - Vn)) > epsilon:
        #print(Vn, Vn1)
        #print(Vn1 - Vn)
        #print(max(Vn1 - Vn))
        count += 1
        Vn = np.copy(Vn1)
        Vn1[0],a0 = T(0, Vn[0], Vn[1], gamma)
        Vn1[1],a1 = T(1, Vn[0], Vn[1], gamma)
        #print("Après changement")
        #print(Vn)
        #print(Vn1)
        #print(Vn1 - Vn)
        #print(max(Vn1 - Vn))

    return Vn, count, a0, a1

gamma_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

def creation_list(gam):
    val0_list = []
    val1_list = []
    act0_list = []
    act1_list = []
    for elt in gam:
        val0_list.append(value_iteration(elt)[0][0])
        val1_list.append(value_iteration(elt)[0][1])
        act0_list.append(value_iteration(elt)[2])
        act1_list.append(value_iteration(elt)[3])
    return val0_list, val1_list, act0_list, act1_list

val0_list, val1_list, act0_list, act1_list = creation_list(gamma_list)

# plot val0_list and val1_list on the same figure in scatter plot
plt.scatter(gamma_list, val0_list, label="Value of the value function for state 0")
plt.scatter(gamma_list, val1_list, label="Value of the value function for state 1")
plt.xlabel("Gamma")
plt.ylabel("Value")
plt.title("Value of the value function for different values of gamma")
plt.legend()

plt.figure()
# plot act0_list and act1_list on the same figure in scatter plot
plt.scatter(gamma_list, act0_list, label="Action to take for state 0")
plt.scatter(gamma_list, act1_list, label="Action to take for state 1")
plt.xlabel("Gamma")
plt.ylabel("Action to take")
plt.title("Action to take for different values of gamma")
plt.legend()
plt.show()
