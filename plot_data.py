# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:27:08 2015

@author: enrico
"""

import pickle
from casadi import *
from casadi.tools import *
import numpy as np
from matplotlib import pyplot as plt
from manipulator_2links import manipulator_2links

sol = pickle.load(open('final.p'))
sol_x = vertcat(sol["X",:,0:2])
sol_dx = vertcat(sol["X",:,2:4])
sol_u = vertcat(sol["U",:])

x = np.array(sol_x).reshape((sol_x.size()/2),2)
dx = np.array(sol_dx).reshape((sol_dx.size()/2),2)
u = np.array(sol_u)

plt.figure(0,[16, 9]); plt.figure;plt.plot(x[:,0], color='r', lw=3.0)
plt.plot(x[:,1], color='g',lw=3.0)
plt.title('q',fontsize=25); plt.xlabel('sample',fontsize=25); plt.ylabel('rad',fontsize=25)
plt.legend(['q1', 'q2'],fontsize=25); plt.grid()

plt.figure(1,[16, 9]); plt.figure;plt.plot(dx[:,0], color='r',lw=3.0)
plt.plot(dx[:,1], color='g',lw=3.0)
plt.title('q_dot',fontsize=25); plt.xlabel('sample',fontsize=25); plt.ylabel('rad/sec',fontsize=25)
plt.legend(['q_dot1', 'q_dot2'],fontsize=25);plt.grid()

plt.figure(2,[16, 9]);plt.axes(xlim=(0, 200), ylim=(-25, 25))
plt.plot(u[:], color='b',lw=3.0)
plt.plot([-20]*200,color='r', lw=1.0)
plt.plot([20]*200,color='r', lw=1.0)
plt.title('u',fontsize=25); plt.xlabel('sample',fontsize=25); plt.ylabel('Nm',fontsize=25)
plt.grid(); 

B = DMatrix(2,1)
B[1] = 1.0
manip = manipulator_2links(B, contacts=True)

Fn = np.array([0]*sol_x.size()).reshape((sol_x.size()/2),2)
for i in range(sol_x.size()/2):
    Fn[i,0] = np.array(vertcat(manip.F_eval([x[i,:], dx[i,:]])))[0][0]
    Fn[i,1] = np.array(vertcat(manip.F_eval([x[i,:], dx[i,:]])))[2][0]

plt.figure(3,[16, 9]); plt.figure;plt.plot(Fn[:,0], color='b',lw=3.0)
plt.plot(Fn[:,1], color='k',lw=3.0)
plt.title('Fn',fontsize=25); plt.xlabel('sample',fontsize=25); plt.ylabel('N',fontsize=25)
plt.legend(['Fn1', 'Fn2'],fontsize=25); plt.grid()