#-*- coding:utf-8 -*-
'''
DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC)
See: http://arxiv.org/abs/1506.04573

Learning algorithm implementation

@author: Pascal Germain -- http://researchers.lille.inria.fr/pgermain/
'''
from kernel import Kernel, KernelClassifier
import numpy as np
from math import sqrt, pi
from numpy import exp, maximum
from scipy.special import erf
from scipy import optimize
from collections import OrderedDict

# Some useful constants
CTE_1_SQRT_2    = 1.0 / sqrt(2.0)
CTE_1_SQRT_2PI  = 1.0 / sqrt(2 * pi)
CTE_SQRT_2_PI   = sqrt(2.0 / pi)

# Some useful functions, and their derivatives
# 此处的x是y⋅w⊤x, erf(x/\sqrt(2)) = 2Φ(x)−1, Φ(x)-CDF 
# 所以定义它干嘛
def gaussian_loss(x): # 1- Φ(x)
    return 0.5 * ( 1.0 - erf(x * CTE_1_SQRT_2) )

def gaussian_loss_derivative(x):
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)

def gaussian_convex_loss(x):
    return maximum( 0.5*(1.0-erf(x*CTE_1_SQRT_2)) , -x*CTE_1_SQRT_2PI+0.5 )

def gaussian_convex_loss_derivative(x):
    x = maximum(x, 0.0)
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)

# 1-p^2-(1-p)^2 = 1-p^2-1-p^2+2p = 2p(1-p)
# erf = 2p-1 -> p = 0.5(erf+1)
# disagreement =-1* 0.5(erf+1)(erf-1)=-1*0.5(erf^2-1) = 0.5*(1-erf^2)
def gaussian_disagreement(x): 
    return 0.5 * ( 1.0 - (erf(x * CTE_1_SQRT_2))**2 )

def gaussian_disagreement_derivative(x):
    return -CTE_SQRT_2_PI * erf(x * CTE_1_SQRT_2) * exp(-0.5 * x**2)

# 两个分类器都出错：JointError(x)=(1−Φ(x))^2
def gaussian_joint_error(x):
    return 0.25 * ( 1.0 - erf(x * CTE_1_SQRT_2) )**2

def gaussian_joint_error_derivative(x):
    return -CTE_1_SQRT_2PI * exp(-0.5 * x**2)  * ( 1.0 - erf(x * CTE_1_SQRT_2) )

JE_SADDLE_POINT_X = -0.5060544689891808
JE_SADDLE_POINT_Y = gaussian_joint_error(JE_SADDLE_POINT_X)
JE_SADDLE_POINT_DX = gaussian_joint_error_derivative(JE_SADDLE_POINT_X)

def gaussian_joint_error_convex(x):
    return maximum( gaussian_joint_error(x), JE_SADDLE_POINT_DX * (x - JE_SADDLE_POINT_X) + JE_SADDLE_POINT_Y)

def gaussian_joint_error_convex_derivative(x):
    return gaussian_joint_error_derivative( maximum(x, JE_SADDLE_POINT_X) )


# Main learning algorithm
class Dalc:
    def __init__(self, B=1.0, C=1.0, convexify=False, nb_restarts=1, verbose=False, nodalc=False, post=False, alpha=None):
        """Pbda learning algorithm.
        B: Trade-off parameter 'B' (source joint error modifier)
        C: Trade-off parameter 'C' (target disagreement modifier)
        convexify: If True, the source loss function is convexified (False by default)
        nb_restarts: Number of random restarts of the optimization process.
        verbose: If True, output informations. Otherwise, stay quiet.
        """       
        self.B = float(B)
        self.C = float(C)
        
        self.nb_restarts = int(nb_restarts)
        self.verbose = bool(verbose)

        if convexify:
            self.source_loss_fct = gaussian_joint_error_convex
            self.source_loss_derivative_fct = gaussian_joint_error_convex_derivative
        else:
            self.source_loss_fct = gaussian_joint_error
            self.source_loss_derivative_fct = gaussian_joint_error_derivative
        self.nodalc = nodalc
        self.post = post
        if(self.post):
            assert(alpha is not None)
            self.alpha = alpha


    def learn(self, source_data, target_data, kernel=None, return_kernel_matrix=False):
        """Launch learning process."""
        if kernel is None: kernel = 'linear'

        if type(kernel) is str:
            kernel = Kernel(kernel_str=kernel)

        if self.verbose: print('Building kernel matrix.')
        data_matrix_dalc = np.vstack((source_data.X, target_data.X))
        label_vector_dalc = np.hstack((source_data.Y, np.zeros(target_data.get_nb_examples())))
        data_matrix_nodalc = np.vstack((source_data.X))
        label_vector_nodalc = np.hstack((source_data.Y))
        if(self.nodalc):
            data_matrix = data_matrix_nodalc
            label_vector = label_vector_nodalc
        else:
            data_matrix = data_matrix_dalc
            label_vector = label_vector_dalc

        kernel_matrix = kernel.create_matrix(data_matrix)
        
        alpha_vector = self.learn_on_kernel_matrix(kernel_matrix, label_vector)

        classifier = KernelClassifier(kernel, data_matrix, alpha_vector)
        if return_kernel_matrix:
            return classifier, kernel_matrix
        else:
            return classifier

    def learn_on_kernel_matrix(self, kernel_matrix, label_vector):  
        """Launch learning process, from a kernel matrix. In label_vector, 0 indicates target examples."""                     
        self.kernel_matrix = kernel_matrix
        self.label_vector = np.array(label_vector, dtype=int)
        self.target_mask = np.array(self.label_vector == 0, dtype=int)
        self.source_mask = np.array(self.label_vector != 0, dtype=int)

        self.nb_examples = len(self.label_vector)
                
        if np.shape(kernel_matrix) != (self.nb_examples, self.nb_examples):
            raise Exception("kernel_matrix and label_vector size differ.")
        
        if(self.nodalc):
            self.margin_factor = (self.label_vector)/np.sqrt(np.diag(self.kernel_matrix))
        else:
            self.margin_factor = (self.label_vector + self.target_mask) / np.sqrt( np.diag(self.kernel_matrix) )
        
        initial_vector = self.label_vector / float(self.nb_examples)
        if(self.post):
            initial_vector[:len(self.alpha)] = self.alpha
        best_cost, best_output = self.perform_one_optimization(initial_vector, 0)
        
        for i in range(1, self.nb_restarts):
            initial_vector = (np.random.rand(self.nb_examples) - 0.5) / self.nb_examples
            cost, optimizer_output = self.perform_one_optimization(initial_vector, i)
            
            if cost < best_cost:
                best_cost = cost
                best_output = optimizer_output        
        
        self.optimizer_output = best_output
        self.alpha_vector = best_output[0]       
        return self.alpha_vector
    
    def perform_one_optimization(self, initial_vector, i):
        """Perform a optimization round."""  
        if self.verbose: print('Performing optimization #' + str(i+1) + '.')
        if self.nodalc:
            self.calc_cost = self.calc_cost_nodalc
            self.calc_gradient = self.calc_gradient_nodalc
        else:
            self.calc_cost = self.calc_cost_dalc
            self.calc_gradient = self.calc_gradient_dalc
        optimizer_output = optimize.fmin_l_bfgs_b(self.calc_cost, initial_vector, self.calc_gradient) 
        cost = optimizer_output[1] 
        
        if self.verbose:
            print('cost value: ' + str(cost))
            for (key, val) in optimizer_output[2].items():
                if key is not 'grad': print(str(key) + ': ' + str(val))                    
    
        return cost, optimizer_output
                           
    def calc_cost_dalc(self, alpha_vector, full_output=False):
        """Compute the cost function value at alpha_vector."""
        kernel_matrix_dot_alpha_vector = np.dot(self.kernel_matrix, alpha_vector)
        margin_vector = kernel_matrix_dot_alpha_vector * self.margin_factor
        
        joint_err_vector = self.source_loss_fct(margin_vector) * self.source_mask
        loss_source = joint_err_vector.sum()

        disagreement_vector = gaussian_disagreement(margin_vector) * self.target_mask
        loss_target = disagreement_vector.sum()

        KL = np.dot(kernel_matrix_dot_alpha_vector, alpha_vector) / 2
               
        cost = loss_source / self.C + loss_target / self.B + KL / (self.B * self.C)

        if full_output:
            return cost, loss_source, loss_target, KL
        else:
            return cost
        
    def calc_gradient_dalc(self, alpha_vector):
        """Compute the cost function gradient at alpha_vector."""
        kernel_matrix_dot_alpha_vector = np.dot( self.kernel_matrix, alpha_vector )
        margin_vector = kernel_matrix_dot_alpha_vector * self.margin_factor

        d_joint_err_vector = self.source_loss_derivative_fct(margin_vector) * self.margin_factor * self.source_mask
        d_loss_source_vector = np.dot(d_joint_err_vector, self.kernel_matrix)
                        
        d_dis_vector = gaussian_disagreement_derivative(margin_vector) * self.margin_factor * self.target_mask
        d_loss_target_vector = np.dot(d_dis_vector, self.kernel_matrix)
                          
        d_KL_vector = kernel_matrix_dot_alpha_vector

        return d_loss_source_vector / self.C + d_loss_target_vector / self.B + d_KL_vector / (self.B * self.C)
        
    
    def calc_cost_nodalc(self, alpha_vector, full_output=False):
        """Compute the cost function value at alpha_vector."""
        kernel_matrix_dot_alpha_vector = np.dot(self.kernel_matrix, alpha_vector)
        margin_vector = kernel_matrix_dot_alpha_vector * self.margin_factor
        
        loss_vec = np.maximum(0, 1 - margin_vector) ** 2
        assert margin_vector.shape[0] == np.sum(self.source_mask), \
        f"margin_vector shape {margin_vector.shape} doesn't match number of source samples {np.sum(self.source_mask)}"

        loss_source = np.sum(loss_vec * self.source_mask)

        KL  = 0.5 * np.dot(kernel_matrix_dot_alpha_vector, alpha_vector)
               
        cost = loss_source / self.C + KL

        if full_output:
            return cost, loss_source, KL
        else:
            return cost
        
    def calc_gradient_nodalc(self, alpha_vector):
        """Compute the cost function gradient at alpha_vector."""
        kernel_matrix_dot_alpha_vector = np.dot( self.kernel_matrix, alpha_vector )
        margin_vector = kernel_matrix_dot_alpha_vector * self.margin_factor

        diff = np.where(margin_vector < 1, -2 * (1 - margin_vector), 0)
        grad_margin = diff * self.margin_factor
        grad_loss = np.dot(self.kernel_matrix, grad_margin)
                          
        d_KL_vector = kernel_matrix_dot_alpha_vector
        
        grad = grad_loss / self.C + d_KL_vector

        return grad
    
    
    
    
    def get_stats(self, alpha_vector=None):
        """Compute some statistics."""
        if alpha_vector is None: alpha_vector = self.alpha_vector
        if(self.nodalc):
            cost, loss_source, KL = self.calc_cost(alpha_vector, full_output=True)
            nb_examples_source = int(np.sum(self.source_mask))
            stats = OrderedDict()
            stats['cost value'] = cost
            stats['loss source'] = loss_source / nb_examples_source
            stats['source loss fct'] = self.source_loss_fct.__name__
            stats['optimizer warnflag'] = self.optimizer_output[2]['warnflag']
            stats['KL'] = KL

        else:
            cost, loss_source, loss_target, KL = self.calc_cost(alpha_vector, full_output=True)
            nb_examples_source = int(np.sum(self.source_mask))
            stats = OrderedDict()
            stats['B'] = self.B
            stats['C'] = self.C
            stats['cost value'] = cost
            stats['loss source'] = loss_source / nb_examples_source
            stats['loss target'] = loss_target / self.nb_examples
            stats['source loss fct'] = self.source_loss_fct.__name__
            stats['KL'] = KL
            stats['optimizer warnflag'] = self.optimizer_output[2]['warnflag']

        return stats

