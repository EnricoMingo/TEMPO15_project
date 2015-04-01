from casadi import *
from numpy import *
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time

class manipulator_2links:
    """
    Ref. Frame:
        y        
        |
        |
        |
        |
        |
        o-------------->x
    """
    def __init__(self, B, d=[1.0, 1.0], m=[1.0, 1.0], I=[0.01, 0.01], damping=0.1, contacts=False ):
        self.q = SX.sym("q",2)
        self.dq = SX.sym("dq",2)
        self.d = d #[m]
        self.l = [self.d[0]/2.0, self.d[1]/2.0] #[m]
        self.m = m #[Kg]
        self.I = I #[Kg*m^2]
        self.g = 9.81 #[m/s^2]
        self.damping = damping*SX.eye(2) #
        
        self.K =5000.0  # spring stiffness
        self.D = 500.0  # spring damping
        
        self.B = B
        self.u = SX.sym("u",B.size2())
        
        self.c = 0.0
        if contacts:
            self.c=1.0 
        
        
        self.H = SX(2,2)
        a1 = self.I[0] + self.m[0]*self.d[0]**2 + self.I[1] + self.m[1]*self.d[1]**2 + self.m[1]*self.l[0]**2
        a2 = self.m[1]*self.l[0]*self.d[1]
        self.H[0,0] = a1 + 2.0*a2*cos(self.q[1])
        a3 = self.I[1] + self.m[1]*self.d[1]**2
        self.H[0,1] = a2*cos(self.q[1]) + a3
        self.H[1,0] = self.H[0,1]
        self.H[1,1] = a3
        self.H_eval = SXFunction([self.q], [self.H])
        self.H_eval.init()
        
        self.C = SX(2,2)
        self.C[0,0] = -2.0*a2*sin(self.q[1])*self.dq[1]
        self.C[0,1] = -a2*sin(self.q[1])*self.dq[1]
        self.C[1,0] = a2*sin(self.q[1])*self.dq[0]
        self.C[1,1] = 0.0
        self.C_eval = SXFunction([self.q, self.dq], [self.C])
        self.C_eval.init()
        
        self.G = SX(2,1)
        a4 = self.g*(self.m[0]*self.d[0]+self.m[1]*self.l[0])
        a5 = self.g*self.m[1]*self.d[1]
        self.G[0] = a4*cos(self.q[0]) + a5*cos(self.q[0] + self.q[1])
        self.G[1] = a5*cos(self.q[0] + self.q[1])
        self.G_eval = SXFunction([self.q], [self.G])
        self.G_eval.init()
        
        self.fk = SX(2,2) #fk = [p1, p2], pi = [xi, yi]'
        self.fk[0,0] = self.d[0]*cos(self.q[0])
        self.fk[1,0] = self.d[0]*sin(self.q[0])
        self.fk[0,1] = self.fk[0,0] + self.d[1]*cos(self.q[0]+self.q[1])
        self.fk[1,1] = self.fk[1,0] +self.d[1]*sin(self.q[0]+self.q[1])
        self.fk_eval = SXFunction([self.q], [self.fk])
        self.fk_eval.init()
        
        self.Jc = vertcat([jacobian(self.fk[:,0], self.q), jacobian(self.fk[:,1], self.q)])

        self.v = mul(self.Jc, self.dq).reshape([2,2])
        self.v_eval = SXFunction([self.q, self.dq], [self.v])
        self.v_eval.init()
        
        #NormalForces:
        # Explicit expressions for normal forces fn
        heaviside = 100.0
        self.Fn = SX(2,1)
        self.Fn[0] = (self.K*self.fk[1,0]+self.D*self.v[1,0])*(-1.0/(1.0+exp(2.0*heaviside*self.fk[1,0])))
        self.Fn[1] = (self.K*self.fk[1,1]+self.D*self.v[1,1])*(-1.0/(1.0+exp(2.0*heaviside*self.fk[1,1])))
        self.Ft = SX(2,1)
        self.Ft[0] = 0.0
        self.Ft[1] = 0.0        
        self.F = SX(4,1)
        self.F[0] = self.Fn[0]
        self.F[1] = self.Ft[0]
        self.F[2] = self.Fn[1]
        self.F[3] = self.Ft[1]
        self.F_eval = SXFunction([self.q, self.dq], [self.F])
        self.F_eval.init()
        
        
        
        self.fd = mul(self.H.inv(), mul(self.B,self.u)-mul(self.C,self.dq)-self.G-mul(self.damping,self.dq)+self.c*mul(self.Jc.T,self.F))
        self.fd_eval = SXFunction([vertcat([self.q, self.dq]), self.u], [vertcat([self.dq, self.fd])])
        self.fd_eval.init()   
		
        self.plotter = { 'figure':None, 'axes':None, 
                         'j0':None, 'j1':None, 
                         'l0':None, 'l1':None,
                         'terrain': None}
        
    def plot(self,q):
        [ee_fk] = self.fk_eval([q])
        j1 = ee_fk[:,0]
        ee = ee_fk[:,1]
        
        if(self.plotter['figure'] is None):
            self.plotter['figure'] = plt.figure(figsize=(16./2,9./2))
            self.plotter['axes'] = plt.axes(xlim=(-4, 4), ylim=(-4.5/2, 4.5/2))
            self.plotter['figure'].show()
            
            self.plotter['j0'] = plt.Circle((0,0),radius=.1,fc='r')
            self.plotter['j1'] = plt.Circle(j1,radius=.1,fc='r')
            self.plotter['l0'] = plt.Line2D((0, j1[0]), (0,j1[1]), lw=5., 
                                 ls='-', marker='.', 
                                 markersize=10, 
                                 markerfacecolor='r', 
                                 markeredgecolor='r', 
                                 alpha=0.5)
            self.plotter['l1'] = plt.Line2D((j1[0],ee[0]), (j1[1],ee[1]), lw=5., 
                                 ls='-', marker='.', 
                                 markersize=10, 
                                 markerfacecolor='r', 
                                 markeredgecolor='r', 
                                 alpha=0.5)
            self.plotter['terrain'] = plt.Line2D((-4,4), (-.05,-.05), 
                                 lw=2., 
                                 ls='-')
                                 
            self.plotter['axes'].add_line(self.plotter['l0'])
            self.plotter['axes'].add_line(self.plotter['l1'])
            if self.c:
                self.plotter['axes'].add_line(self.plotter['terrain'])
            
            mencoderWriter = animation.writers['mencoder']
            self.plotter['writer'] = mencoderWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        
        else:
            self.plotter['l0'].set_data(((0, j1[0]), (0,j1[1])))
            self.plotter['l1'].set_data(((j1[0],ee[0]), (j1[1],ee[1])))
            
        plt.draw()
        return [self.plotter['l0'], self.plotter['l1']]
        
    def plotTraj(self,qTraj, t=.001, fileName=None):
        
        self.plot(qTraj[0,:])
        p = lambda i : self.plot(qTraj[i,:])
        anim = animation.FuncAnimation(self.plotter['figure'], 
                                       p,
                                       frames=qTraj.shape[0], 
                                       interval=t*1000.,
                                       blit=True)
        plt.draw()
        
        if fileName is not None:
            anim.save(fileName, writer=self.plotter['writer'])
   
        
if __name__=='__main__':
    B = DMatrix(2,1)
    B[1] = 1.0
    manip = manipulator_2links(B,d=[1.0, 1.0],contacts=True)    
    print "l:", manip.l 
    print "d:", manip.d  
    print "m:", manip.m
    print "I:", manip.I
    print "H:", manip.H
    q_eval = [pi/4., 0.0]
    print "H_eval:", manip.H_eval([q_eval]) 
    print "C:", manip.C
    dq_eval = [0.0, 0.0]
    print "C_eval:", manip.C_eval([q_eval, dq_eval])
    print "G:", manip.G
    print "G_eval:", manip.G_eval([q_eval])
    print "fk:", manip.fk
    print "fk_eval:", manip.fk_eval([q_eval])
    print "fd:", manip.fd
    u_eval = [0.0]
    print "fd_eval:", manip.fd_eval([vertcat([q_eval, dq_eval]), u_eval])
    print "J':", manip.Jc.T
    print "F:", manip.F_eval([q_eval, dq_eval])
    print "v:", manip.v
    
    #SIMULATION 
    intg = simpleIRK(manip.fd_eval, 10)  
    intg.init()
    N = 5000
    trj = DMatrix(N,2)
    for i in range(N):    
        h_test = 0.01;
        [x_next] = intg([vertcat([q_eval, dq_eval]), u_eval, h_test]);
        q_eval = x_next[0:2]
        dq_eval = x_next[2:4] 
        
        #print "q_eval:", q_eval 
        trj[i,0] = q_eval[0]
        trj[i,1] = q_eval[1] 
        
        #print "F:", manip.F_eval([q_eval]), "for", manip.fk_eval([q_eval])
        
    manip.plotTraj(trj,fileName='ex.mp4')
    
	
