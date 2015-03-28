from casadi import *
from casadi.tools import *
from pylab import *
from manipulator_2links import manipulator_2links

"""
       NOTE: if you use spyder,
           make sure you open a Python interpreter
                 instead of an IPython interpreter
           otherwise you wont see any plots
"""


N = 100     # Control discretization
T = 10.    # End time
h = T/N
M = 10      # Number of RK4 steps

# Declare variables (use scalar graph)
B = DMatrix(2,1)
B[1] = 1.0
manip = manipulator_2links(B)

u  = manip.u    # control
x  = vertcat([manip.q, manip.dq])  # states

# Formulate the ODE

f = manip.fd_eval

# Discrete time dynamics
F = simpleRK(f, M)
F.init()

# Define NLP variables
W = struct_symMX([
      (
        entry("X",shape=(4,1),repeat=N+1),
        entry("U",shape=(1,1),repeat=N)
      )
])

w = MX(W)

# NLP constraints
g = []

# Build up a graph of integrator calls
for k in range(N):
    # Call the integrator
    [x_next_k] = F([ W["X",k], W["U",k], h ])

    # Append continuity constraints
    g.append(x_next_k - W["X",k+1])

# Concatenate constraints
g = vertcat(g)

# Objective function
R = vertcat([vertcat(W['X',2:4]),vertcat(W['U'])])
obj = mul(R.T, R)

# Create an NLP solver object
nlp = MXFunction(nlpIn(x=W),nlpOut(f=obj,g=g))
nlp_solver = NlpSolver("ipopt", nlp)
nlp_solver.setOption("linear_solver", "mumps")
nlp_solver.init()

# All constraints are equality constraints in this case
nlp_solver.setInput(0, "lbg")
nlp_solver.setInput(0, "ubg")

# Construct and populate the vectors with
# upper and lower simple bounds
w_min = W(-inf)
w_max = W( inf)

# Control bounds
w_min["U",:] = -10
w_max["U",:] = 10

w_k = W(0)
ts = linspace(0,T,N+1)

t = 0
x_0 = array([-pi/2,0,0,0])
x_current = x_0

x_final = [[pi/2,0,0,0]] #upright!

w_min['X',N] = x_final
w_max['X',N] = x_final

w_min["X",0] = x_current
w_max["X",0] = x_current

# Pass data to NLP solver
nlp_solver.setInput(w_k,"x0")
nlp_solver.setInput(w_min,"lbx")
nlp_solver.setInput(w_max,"ubx")
   
# Solve the OCP
nlp_solver.evaluate()
    
# Extract from the solution the first control
sol = W(nlp_solver.getOutput("x"))
u_nmpc = sol["U",0]

manip.plotTraj(np.array(vertcat(sol["X",:,0:2])).reshape(N+1,2),t=T/N)

"""
# Simulate the system with this control
[x_current] = F([x_current, u_nmpc, h])
  
t += T/N
# Shift the time to have a better initial guess
# For the next time horizon
w_k["X",:-1] = sol["X",1:]
w_k["U",:-1] = sol["U",1:]
w_k["X",-1] = sol["X",-1]
w_k["U",-1] = sol["U",-1]
"""