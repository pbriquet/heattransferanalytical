import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm

class DirichletRectangular:
    """ 
    This is a class for analytical solution of Diffusion in d dimensions with constant Dirichlet condition at boundaries (1,1,1), and zero flux at center (0,0,0) (symmetry condition).
    \nAt Init, the user can define the number of dimensions (dim=3), number of eigenvalues at each dimnesion (n_eigen=100), and scale of body (a,b,c) (scale=[1.0,1.0,1.0])
    \nBefore calculating the solution, the user must fix a position with prepare_solution(coordinate=[0.0,0.0,0.0])
    \n- solution(t):\t Calculate for each instant. The function doesn't accept "t" arrays yet.
    \n- average(t):\t Calculate the average field for each instant.
    \n- solution_center(t):\t Calculate the field at center position for each instant.
    """

    @staticmethod
    def copy(solution):
        if(isinstance(solution,DirichletRectangular)):
            tmp = DirichletRectangular(dim=solution.dim,n_eigen=1,scale=solution.scale,zerofluxcenter=solution.zerofluxcenter)
            tmp.n_eigen = solution.n_eigen
            tmp.mesh_l = solution.mesh_l
            tmp.mesh_scaled_l = solution.mesh_scaled_l
            tmp.mesh_A = solution.mesh_A
            tmp.alpha = solution.alpha
            tmp.mesh_A2 = solution.mesh_A2
            return tmp
        else:
            return Exception('Not the same class')
    def __init__(self,dim=3,n_eigen=100,scale=[1.0,1.0,1.0],zerofluxcenter=True):
        self.zerofluxcenter = zerofluxcenter
        self.n_eigen = n_eigen  # Number of eigenvalues for solution (equal for each dimension) n^dim = number of coefficients
        self.dim = dim          # 1D, 2D, 3D
        self.scale = scale      # a, b, c Rectangular Lengths
        self._calculate_coefficients()
    def _calculate_coefficients(self):
        if(self.zerofluxcenter): 
            # With dT/dx = 0 at center, solution is given by cos(λ)=0
            self.lambs = np.array([(m+0.5)*np.pi for m in range(self.n_eigen)]) # λ', β', γ' arrays are the same
        else:
            # With T(0) = 0 and T(1) = 0, solution is given by sin(λ)=0
            self.lambs = np.array([m*np.pi for m in range(self.n_eigen)])
        arrays = [self.lambs for k in range(self.dim)] # Just putting in a list depending on dim.
        # λ = (a/a)λ', β = (a/b)β', γ = (a/c)γ' arrays
        scaled_lambs = [self.lambs*self.scale[0]/self.scale[k] for k in range(self.dim)] 
        # Meshgrid of λ'i, β'j, γ'k
        self.mesh_l = np.array(np.meshgrid(*arrays,indexing='ij'))
        # Meshgrid of λi, βj, γk
        self.mesh_scaled_l = np.array(np.meshgrid(*scaled_lambs,indexing='ij'))
        # Calculating A_ijk = <f,g_ijk>/<g_ijk,g_ijk> with f = 1 and g_ijk = cos(λ'i*x)*cos(β'j*y)*cos(γ'k*z)
        self.mesh_A = np.power(2,self.dim)*np.prod(np.sin(self.mesh_l)/self.mesh_l,axis=0)
        # α_ijk = λi^2 + βj^2 + γk^2
        self.alpha = np.sum(np.power(self.mesh_scaled_l,2),axis=0)
        # Calculating A^2_ijk. Useful for average field.
        self.mesh_A2 = np.power(self.mesh_A,2)
    def prepare_position(self,coordinate):
        self.X = np.prod(np.cos(np.array([self.mesh_l[i]*coordinate[i] for i in range(self.dim)])),axis=0)
        self.prepared_coordinate = True
    # Solution is given by: sum_i(sum_j(sum_k(A_ijk*cos(λ'i*x)*cos(β'j*y)*cos(γ'k*z)*exp(-α_ijk*t))))
    def solution(self,time):    # Add a way to receive arrays
        tmp = np.sum(self.mesh_A*self.X*np.exp(-self.alpha*time))
        return tmp
    # Average is given by: 1/2^d*sum_i(sum_j(sum_k(A^2_ijk*exp(-α_ijk*t))))
    def average(self,time):
        temp = np.sum(self.mesh_A2*np.exp(-self.alpha*time))/np.power(2,self.dim)
        return temp
    # Solution at center is given by: sum_i(sum_j(sum_k(A_ijk*exp(-α_ijk*t))))
    def solution_center(self,time):
        tmp = np.sum(self.mesh_A*np.exp(-self.alpha*time))
        return tmp

if __name__=='__main__':
    number_of_positions = 5
    analytical = DirichletRectangular(dim=1,n_eigen=100)
    a_array = [DirichletRectangular.copy(analytical) for i in range(number_of_positions)]
    b_array = [{'X':[],'Y':[]} for i in range(number_of_positions)]
    for i in range(number_of_positions):
        a_array[i].prepare_position([i/number_of_positions])
    t = np.linspace(1e-4,0.7,num=500)
    for _t in t:
        for k,v in enumerate(b_array):
            v['X'].append(_t)
            v['Y'].append(a_array[k].solution(_t))

    fig = plt.figure()
    ax =fig.add_subplot(111)
    for k,v in enumerate(b_array):
        ax.plot(v['X'],v['Y'])
    plt.show()