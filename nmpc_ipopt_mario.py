from casadi import *
from casadi.tools import *
from pylab import *
from manipulator_2links import manipulator_2links
from time import *

N = 200     # Control discretization
T = 2.    # End time
h = T/float(N)
M = 1    # Number of IRK4 steps

# Declare variables (use scalar graph)
B = DMatrix(2,1)
B[1] = 1.0
manip = manipulator_2links(B, contacts=True)

u  = manip.u    # control
x  = vertcat([manip.q, manip.dq])  # states

# Formulate the ODE

f = manip.fd_eval

# Discrete time dynamics
#F = simpleRK(f, M)
F = simpleIRK(f, M, 2, "radau")
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

#Add fk constraint
#for k in range(N):
#    g.append(manip.fk_eval([W["X",k][0:2]])[0][1,0])
    
#for k in range(N):
#    g.append(manip.fk_eval([W["X",k][0:2]])[0][1,1])

##Add v constraint
#for k in range(N):
#    g.append(manip.v_eval([W["X",k][0:2],W["X",k][2:4]])[0][1,0])
#    
#for k in range(N):
#    g.append(manip.v_eval([W["X",k][0:2],W["X",k][2:4]])[0][1,1])


# Concatenate constraints
g = vertcat(g)

# Objective function
R = vertcat([vertcat(W['X',2:4]),vertcat(W['U'])])
obj = mul(R.T, R)
obj += 1000*mul( vertcat(W['X',:2]- np.array([pi/2.,0])).T, vertcat(W['X',:2]- np.array([pi/2.,0])))

# Create an NLP solver object
nlp = MXFunction(nlpIn(x=W),nlpOut(f=obj,g=g))
nlp_solver = NlpSolver("ipopt", nlp)
nlp_solver.setOption("linear_solver", "mumps")
nlp_solver.init()

# All constraints are equality constraints in this case
#g_min = DMatrix([0.]*6*N)
#g_min[4*N::] = -0.001;
#g_min[6*N::] = -1.
#g_max = DMatrix([0.0]*6*N)
#g_max[4*N::] = inf;
#g_max[6*N::] = 1.
nlp_solver.setInput(0.0, "lbg")
nlp_solver.setInput(0.0, "ubg")

# Construct and populate the vectors with
# upper and lower simple bounds
w_min = W(-inf)
w_max = W( inf)

# Control bounds
w_min["U",:] = -20.
w_max["U",:] = 20.

w_k = W(0.)
x_current = array([0.,0.,0.,0.])

x_final = [[pi/2.,0.,0.,0.]] #upright!
#w_k["U"] = 5.
#w_k["X",:] = x_final*(N+1)
w_min['X',-1] = x_final
w_max['X',-1] = x_final

#t = 0
for i in range(1):

    w_min["X",0] = x_current
    w_max["X",0] = x_current

    # Pass data to NLP solver
    nlp_solver.setInput(w_k,"x0")
    nlp_solver.setInput(w_min,"lbx")
    nlp_solver.setInput(w_max,"ubx")
   
    # Solve the OCP
    tic = time()
    nlp_solver.evaluate()
    toc = time()
    print "solver needed:",toc-tic,"[s]"
    
    
    # Extract from the solution the first control
    sol = W(nlp_solver.getOutput("x"))
    #u_nmpc = sol["U",0]
    
    # Simulate the system with this control
    #[x_current] = F([x_current, u_nmpc, h])
  
    #t += T/N
    
    # Shift the time to have a better initial guess
    # For the next time horizon
    #w_k["X",:-1] = sol["X",1:]
    #w_k["U",:-1] = sol["U",1:]
    #w_k["X",-1] = sol["X",-1]
    #w_k["U",-1] = sol["U",-1]

manip.plotTraj(np.array(vertcat(sol["X",:,0:2])).reshape(N+1,2),t=T/N)