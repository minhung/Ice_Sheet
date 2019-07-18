#from dolfin import *

from fenics import *

def E(u):
    return 0.5*( grad(u) + grad(u).T )
def tau(u,n):
    return u - inner(u,n)*n

def Dfun(etensor):
    return inner(etensor,etensor)

# Define parameters
alpha = 4.0
gamma = 8.0
nu = 1.0
mu =2.0
theta = 1.0 #SIP
Kf = 2 #degree

# FIXME: Make mesh ghosted
#parameters["ghost_mode"] = "shared_facet"

# Define class marking Dirichlet boundary (x = 0 or x = 1)
class DirichletBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0]*(1 - x[0]), 0)

# Define class marking Neumann boundary (y = 0 or y = 1)
class NeumanBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[1]*(1 - x[1]), 0)
    
# Create mesh and define function space
#nu = 1
mesh = UnitSquareMesh(48, 48)
P1dv = VectorElement("DG", triangle, 2); P1d = FiniteElement("DG", triangle, 2)
P1dP1d = MixedElement([P1dv, P1d]); W = FunctionSpace(mesh, P1dP1d)
P0 = FiniteElement("DG", triangle, 0); Q0 = FunctionSpace(mesh, P0)

# Define test and trial functions
(v,q) = TestFunctions(W)
(u,p) = TrialFunctions(W)


# Define normal vector and mesh size
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

# Define the source term f, Dirichlet term u0 and Neumann term g
#uSex = Expression(('-exp(x[0])*(x[1]*cos(x[1])+sin(x[1]))', \
#               'exp(x[0])*x[1]*sin(x[1])'),degree=5)
#f = Expression(('-100.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)', \
#               '-100.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)'),degree=3)
#pSex = Expression('exp(x[0])*sin(x[1])', degree=5)

#f = Expression(('-exp(x[0])*sin(x[1])', \
#               '-exp(x[0])*cos(x[1])'),degree=3)
#f = Expression(('0','0'),degree=3)

#Example 1
uSex = Expression(('-exp(x[0])*(x[1]*cos(x[1])+sin(x[1]))', \
               'exp(x[0])*x[1]*sin(x[1])'),degree=5)
#f = Expression(('-100.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)', \
#               '-100.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)'),degree=3)
pSex = Expression('2.0*exp(x[0])*sin(x[1])-2.0*(1.0-exp(1.0))*(cos(1.0)-1.0)/3.0', degree=5)

f0 = Expression('-exp(x[0])', degree=5)
fD = Expression('(1.0+2.0*exp(2.0*x[0])*(1.0+x[1]*x[1]))*(1.0+2.0*exp(2.0*x[0])*(1.0+x[1]*x[1]))',\
               degree=5)
f2 = Expression(('sin(x[1])*(1.0+exp(2.0*x[0])*(6.0+2.0*x[1]*x[1]))+cos(x[1])*exp(2.0*x[0])*x[1]*(8.0+4.0*x[1]*x[1])',\
                 'cos(x[1])*(1.0+exp(2.0*x[0])*(6.0+2.0*x[1]*x[1]))-sin(x[1])*exp(2.0*x[0])*x[1]*(8.0+4.0*x[1]*x[1])'),\
                 degree=5)


#fnew = -div(nu* E(Constant(1.0, cell = mesh.ufl_cell()) *uSex))\
#+ grad(Constant(1.0, cell = mesh.ufl_cell()) *pSex)

#u0 = Expression('x[0] + 0.25*sin(2*pi*x[1])', degree=3)
#g = Expression('(x[1] - 0.5)*(x[1] - 0.5)', degree=3)

# Mark facets of the mesh
boundaries = MeshFunction('size_t', mesh, 0)
NeumanBoundary().mark(boundaries, 2)
DirichletBoundary().mark(boundaries, 1)

# Define outer surface measure aware of Dirichlet and Neumann boundaries
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# error computation
V = FunctionSpace(mesh, P1dv); Q = FunctionSpace(mesh, P1d);

# Define variational problem

# Initial Setup
#Ah(uv)
Ahuv = \
mu*inner(E(u),E(v))*dx \
-mu*inner(avg(E(u))*n('+'),jump(v))*dS - mu*inner(E(u)*n,v)*ds \
+theta*mu*inner(avg(E(v))*n('+'),jump(u))*dS +theta*mu*inner(E(v)*n,u)*ds \
+gamma/h_avg*inner(jump(u),jump(v))*dS + gamma/h*inner(u,v)*ds \

#Bh(vp)
Bhvp= - p*div(v)*dx + avg(p)*inner(jump(v),n('+'))*dS + p*inner(v,n)*ds

#Bh(uq)
Bhuq = - q*div(u)*dx + avg(q)*inner(jump(u),n('+'))*dS + q*inner(u,n)*ds

# a_Stokes
a = Ahuv + Bhvp - Bhuq

# RHS
#inner(fnew,v)*dx \
#L =  inner(-div(nu*E(uSex))+grad(pSex),v)*dx  \
#L =  inner(-div(nu* E(Constant(1.0, cell = mesh.ufl_cell()) *uSex))\
#+ grad(Constant(1.0, cell = mesh.ufl_cell()) *pSex),v)*dx\

L=inner(f0*fD*f2,v)*dx \
+ mu*inner(E(v)*n,uSex)*ds \
- q*inner(uSex,n)*ds \
+ gamma/h*inner(uSex,v)*ds #

# Solve problem
w = Function(W)
solve(a == L, w)
(ui,pi) = w.split()

# error computation
uSer = project(ui-uSex, V); pSer = project(pi-pSex, Q)
uS_L2er = sqrt( assemble(inner(uSer,uSer)*dx) ); uS_H1er = sqrt( assemble( inner(grad(uSer),grad(uSer))*dx ) ); pS_L2er = sqrt( assemble(pSer*pSer*dx) )
print ("\n")
print ("|u-uh|_{L2,S} \t |u-uh|_{H1,S} \t |p-ph|_{L2,S}")
print ("%g \t %f \t %f" % (uS_L2er, uS_H1er, pS_L2er))
print ("\n")


# Iteration
# mu = 2+1/(1+E:E)

count = 0
while count < 10:
    count += 1
#Ah(uv)
    Ahuv = \
    (2.0+1.0/(1.0+inner(E(ui),E(ui))))*inner(E(u),E(v))*dx \
    -(2.0+1.0/(1.0+inner(avg(E(ui)),avg(E(ui)))))*inner(avg(E(u))*n('+'),jump(v))*dS \
    -(2.0+1.0/(1.0+inner(E(ui),E(ui))))*inner(E(u)*n,v)*ds \
    +theta*(2.0+1.0/(1.0+inner(jump(ui,n),jump(ui,n))/(h_avg*h_avg)))*inner(avg(E(v))*n('+'),jump(u))*dS \
    +theta*(2.0+1.0/(1.0+inner(ui,ui)/(h*h)))*inner(E(v)*n,u)*ds \
    +gamma/h_avg*inner(jump(u),jump(v))*dS + gamma/h*inner(u,v)*ds \

#Bh(vp)
    Bhvp= - p*div(v)*dx + avg(p)*inner(jump(v),n('+'))*dS + p*inner(v,n)*ds

#Bh(uq)
    Bhuq = - q*div(u)*dx + avg(q)*inner(jump(u),n('+'))*dS + q*inner(u,n)*ds

# a_Stokes
    a = Ahuv + Bhvp - Bhuq

# RHS
    L=inner(f0*fD*f2,v)*dx \
    + (2.0+1.0/(1.0+inner(E(ui),E(ui))))*inner(E(v)*n,uSex)*ds \
    - q*inner(uSex,n)*ds \
    + gamma/h*inner(uSex,v)*ds #

# Solve problem
    w = Function(W)
    solve(a == L, w)
    (ui,pi) = w.split()

    usol = ui
    psol = pi

# additive const
    p0 = psol((0,0)); print ("p0 =", p0); psol = psol-p0

# error computation
#V = FunctionSpace(mesh, P1dv); Q = FunctionSpace(mesh, P1d);
    uSer = project(usol-uSex, V); pSer = project(psol-pSex, Q)
    uS_L2er = sqrt( assemble(inner(uSer,uSer)*dx) ); uS_H1er = sqrt( assemble( inner(grad(uSer),grad(uSer))*dx ) ); pS_L2er = sqrt( assemble(pSer*pSer*dx) )
    print ("Iteration:%d:\n",count)
    print ("|u-uh|_{L2,S} \t |u-uh|_{H1,S} \t |p-ph|_{L2,S}")
    print ("%g \t %f \t %f" % (uS_L2er, uS_H1er, pS_L2er))
    print ("\n")

# h_avg =avg(h)
# This is v^+ - v^-
#jump(v)
# This is (v^+ - v^-) n
#jump(v, n)


(v,q)=TestFunctions(W)
(utest,p) = TrialFunctions(W)
test = sqrt(inner(u,u))*inner(E(utest),E(v))*dx
test2= sqrt(inner(jump(u,n),jump(u,n)))*inner( avg(E(utest))*n('+'),jump(v) ) *dS
test3= Dfun(E(u))*inner(E(utest),E(v))*dx
atest = test+test2+test3