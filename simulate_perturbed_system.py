# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:30:18 2015

@author: arocchi
"""
from casadi import *
from casadi.tools import *
from pylab import *
from manipulator_2links import manipulator_2links
from time import *
import pickle

N = 200     # Control discretization
T = 2.    # End time
h = T/float(N)
M = 1    # Number of IRK4 steps

B = DMatrix(2,1)
B[1] = 1.0
manip_perturbed = manipulator_2links(B, contacts=True, K = 4900.0)

#u  = manip.u    # control
#x  = vertcat([manip.q, manip.dq])  # states

F_sim = simpleIRK(manip_perturbed.fd_eval, M, 2, "radau")
F_sim.init()

u = pickle.load(open('nmpc_u_perturbed_4900.p'))
#u = pickle.load(open('nmpc_u_perturbed_4800.p'))
x_0 = array([0.,0.,0.,0.])

x_k = x_0
q_all = []
q_all.append(np.array(x_k[0:2]).reshape(1,2)[0])
for k in range(N):
    [x_next] = F_sim([x_k, u[k], h])
    q_all.append([x_next[0],x_next[1]])
    x_k = x_next

manip_perturbed.plotTraj(np.array(q_all),t=T/N,fileName = 'swingup_perturbed.mp4')