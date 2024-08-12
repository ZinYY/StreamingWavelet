'''
This is an implementation for Streaming Wavelet Operator,
which sequentially applies wavelet transform to an online sequence efficiently.
    ** Time complexity: O(k * log(T)) for each update,
       where k is the order, and T is the sequence length.
    ** Space complexity: O(k * log(T)).
    
Reference:
Qian et al., Efficient Non-stationary Online Learning by Wavelets
with Applications to Online Distribution Shift Adaptation.
In Proceedings of the 41st International Conference on Machine Learning (ICML 2024).
'''

import copy
from .MakeCDJVFilter import MakeCDJVFilter
from .wavelets_coeff import \
    (db2, db3, db4, db5, db6, db7, db8, db9, db10,
     db11, db12, db13, db14, db15, db16, db17, db18, db19, db20,
     sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10,
     sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20,
     bior1_1, bior1_3, bior1_5, bior2_2, bior2_4, bior2_6, bior2_8, bior3_1, bior3_3, bior3_5,
     bior3_7, bior3_9, bior4_4, bior5_5, bior6_8,
     coif1, coif2, coif3, coif4, coif5,
     dmey, haar)
import math
import numpy as np


def cdjv_dyad_down(bhi, F, LEF, REF):
    n = bhi.shape[0]
    dim = bhi.shape[1]
    N = len(F) // 2
    kernel = F[:, 1].reshape(-1)
    
    y = np.convolve(np.sum(np.flip(bhi), axis=1), kernel).reshape(-1, 1).repeat(dim, axis=1)
    
    # y = np.apply_along_axis(lambda x: np.convolve(x, kernel), 0, np.flip(bhi))
    y = y[:, 0: dim]
    
    # y = convolve(bhi, np.flip(F).reshape(-1, 1), mode='full', method='auto')  # scipy's convolve
    
    LEDGE = bhi[:3 * N - 1, :][::-1, :]
    REDGE = bhi[-1:-3 * N:-1, :]
    LEvals = LEF @ LEDGE
    REvals = REF @ REDGE
    blo = np.zeros((n // 2, bhi.shape[1]))
    blo[:N, :] = LEvals
    blo[-N:, :] = REvals[::-1, :]
    blo[N:n // 2 - N, :] = y[3 * N:3 * N + 1 + 2 * (n // 2 - 2 * N - 1):2, :]
    return blo


class StreamingWavelet_order1():
    '''
    This class is used to sequentially apply wavelet transform to
        an online sequence efficiently (with lazy update & bitwise operation).
    Time Complexity: O(k * log T) every update (add_signal), where k is the order
    Storage Complexity: a total O(k * log T), where k is the order
    
    This code implements streaming update for the following wavelets:
        db2, db3, db4, db5, db6, db7, db8, db9, db10,
        db11, db12, db13, db14, db15, db16, db17, db18, db19, db20,
        sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10,
        sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20,
        bior1_1, bior1_3, bior1_5, bior2_2, bior2_4, bior2_6, bior2_8, bior3_1, bior3_3, bior3_5,
        bior3_7, bior3_9, bior4_4, bior5_5, bior6_8,
        coif1, coif2, coif3, coif4, coif5, dmey, haar
    '''
    
    def __init__(self, dim, max_length, order, get_coeff=False, axis=-1, verbose=False):
        self.order = order
        self.get_coeff = get_coeff
        self.axis = axis
        self.max_length = max_length
        self.total_layers = (int(math.log2(self.max_length - 0.5)) + 1) + 1
        self.dim = dim
        self.verbose = verbose
        
        # wavelets of different orders
        self.wavelets_coeff = haar.Haar()
        
        self.wavelets_coeff.decompositionLowFilter = \
            np.array(self.wavelets_coeff.decompositionLowFilter).reshape(-1, 1).repeat(dim, axis=1)
        self.wavelets_coeff.decompositionHighFilter = \
            np.array(self.wavelets_coeff.decompositionHighFilter).reshape(-1, 1).repeat(dim, axis=1)
        self.wavelets_coeff_length = self.wavelets_coeff.__motherWaveletLength__
        
        self.seq_legth = 0
        self.total_2_norm = 0.0
        self.coeff_arrs = [None] * (self.total_layers - 1)
        
        self.layer_arr = np.array(range(self.total_layers + 1))
        self.sum_arrs = [None] * self.total_layers
        for i in range(self.total_layers):
            self.sum_arrs[i] = []
        
        if self.get_coeff == True:
            self.all_coeff_arrs = []  # Calculate all wavelet coefficients, need O(T) storage space
    
    def get_norm(self):
        if self.seq_legth <= 1:
            return 0.0
        
        # Get the full norm
        full_norm = self.total_2_norm
        for i in range(self.total_layers):
            max_not_none_idx = 0
            if len(self.sum_arrs[i]) == 0: break
            for j in range(0, len(self.sum_arrs[i])):
                max_not_none_idx = j
                if np.any(self.sum_arrs[i][j][1]) == None:
                    break
            
            # Use convolution to speed up
            if max_not_none_idx <= 1: break
            tmp_vec = np.array(self.sum_arrs[i][1:max_not_none_idx]).reshape(-1, self.dim)
            
            full_norm += np.linalg.norm(np.sum(
                np.convolve(tmp_vec, self.wavelets_coeff.decompositionHighFilter, "full")
                [:-tmp_vec.shape[0] - 1], axis=0))
        
        return math.sqrt(full_norm)
    
    def get_log2position(self, position):
        # get the update position of each layer,
        # use bitwise operation to speed up.
        update_idx = np.bitwise_and(np.right_shift(position, self.layer_arr), 1)
        update_idx[position < np.left_shift(1, self.layer_arr)] = -1
        return update_idx
    
    def add_signal(self, element):
        # element = element.numpy()
        update_idx_arrs = self.get_log2position(self.seq_legth)
        
        if self.verbose:
            print(self.seq_legth, update_idx_arrs)
        
        none_num = 0
        for i in range(self.total_layers):
            if update_idx_arrs[i] == -1:
                none_num += 1
                update_idx_arrs[i] = 0
            
            if none_num == 2: break
            
            if update_idx_arrs[i] == 0 and (len(self.sum_arrs[i]) == 0 or self.sum_arrs[i][-1][0] is not None):
                self.sum_arrs[i].append([])
                self.sum_arrs[i][-1] = [None] * 2
            
            if i == 0:
                self.sum_arrs[i][-1][update_idx_arrs[i]] = element
            
            if len(self.sum_arrs[i]) > self.order:
                self.sum_arrs[i].pop(0)
            
            if i > 0:
                if update_idx_arrs[i] != -1 and self.sum_arrs[i - 1][-1][1] is not None and len(self.sum_arrs[i - 1]) == self.order:
                    # if self.sum_arrs[i - 1][-1][1] is not None and len(self.sum_arrs[i]) == self.order:
                    # print('Update Sum Arr!')
                    self.sum_arrs[i][-1][update_idx_arrs[i]] = \
                        np.sum(np.array(self.sum_arrs[i - 1]).reshape(-1, self.dim)
                               * self.wavelets_coeff.decompositionLowFilter, axis=0)
            
            if self.verbose:
                print('sum_arrs[{}]:'.format(i), len(self.sum_arrs[i]), self.sum_arrs[i])
            
            if update_idx_arrs[i] == 1 and self.sum_arrs[i][-1][1] is not None and len(self.sum_arrs[i]) == self.order:
                # if self.sum_arrs[i][-1][1] is not None and len(self.sum_arrs[i]) == self.order:
                # Update coeff_arrs
                self.total_2_norm += \
                    np.linalg.norm(np.sum(np.array(self.sum_arrs[i]).reshape(-1, self.dim)
                                          * self.wavelets_coeff.decompositionHighFilter, axis=0)) ** 2  # update 2-norm sequentially
                if self.get_coeff and len(self.sum_arrs[i]) == self.order:
                    # self.all_coeff_arrs.append(self.sum_arrs[i][0] - self.sum_arrs[i][1])
                    self.all_coeff_arrs.append(np.sum(np.array(self.sum_arrs[i]).reshape(-1, self.dim)
                                                      * self.wavelets_coeff.decompositionHighFilter, axis=0))
                if self.verbose:
                    print(i, 'coeff_arrs', self.all_coeff_arrs)
        
        self.seq_legth += 1
    
    def reinit(self):
        self.seq_legth = 0
        self.total_2_norm = 0.0
        self.coeff_arrs = [None] * (self.total_layers - 1)
        
        self.layer_arr = np.array(range(self.total_layers + 1))
        self.sum_arrs = [None] * self.total_layers
        for i in range(self.total_layers):
            self.sum_arrs[i] = []
        
        if self.get_coeff == True:
            self.all_coeff_arrs = []  # Calculate all wavelet coefficients, need O(T) storage space


class StreamingWavelet():
    '''
    This class is used to sequentially apply wavelet transform to
        an online sequence signal efficiently (with lazy update & bitwise operation).
        
        Time Complexity: O(k * log T) every update (add_signal), where k is the order
        Storage Complexity: a total O(k * log T), where k is the order
    '''
    
    def __init__(self, dim, max_length, order, get_coeff=False, axis=-1, verbose=False):
        if order == 1:  # recover the haar wavelet
            self.__class__ = StreamingWavelet_order1
            self.__init__(dim, max_length, order, get_coeff, axis, verbose)
        elif order == 2 or order == 3:
            self.order = order
            self.get_coeff = get_coeff
            self.axis = axis
            self.max_length = max_length
            self.total_layers = (int(math.log2(self.max_length - 0.5)) + 1) + 1
            self.dim = dim
            self.verbose = verbose
            
            # wavelets of different orders
            if self.order == 1:
                self.wavelets_coeff = haar.Haar
            elif self.order > 20:
                raise NotImplementedError
            else:
                self.wavelets_coeff = eval('db{}.Daubechies{}'.format(str(self.order), str(self.order)))
            
            # self.wavelets_coeff.decompositionLowFilter = \
            #     np.array(self.wavelets_coeff.decompositionLowFilter).reshape(-1, 1).repeat(dim, axis=1)
            # self.wavelets_coeff.decompositionHighFilter = \
            #     np.array(self.wavelets_coeff.decompositionHighFilter).reshape(-1, 1).repeat(dim, axis=1)
            
            self.HPF, self.LHPEF, self.RHPEF = MakeCDJVFilter('HighPass', order)  # Wavelet function
            self.LPF, self.LLPEF, self.RLPEF = MakeCDJVFilter('LowPass', order)  # Scale function
            self.LPREMAT, self.RPREMAT, _ = MakeCDJVFilter('PreCondition', order)  # Edge function in CDJV
            self.HPF = self.HPF.tolist()
            self.LPF = self.LPF.tolist()
            # self.wavelets_coeff_length = self.wavelets_coeff.__motherWaveletLength__
            self.wavelets_coeff_length = len(self.HPF)
            
            self.LPF = np.array(self.LPF).reshape(-1, 1).repeat(dim, axis=1)
            self.HPF = np.array(self.HPF).reshape(-1, 1).repeat(dim, axis=1)
            
            self.seq_legth = 0
            self.total_2_norm = 0.0
            self.coeff_arrs = [None] * (self.total_layers - 1)
            
            self.left_edge = None
            self.right_edge = None
            self.left_mat = None
            self.right_mat = None
            
            self.layer_arr = np.array(range(self.total_layers + 1))
            self.sum_arrs = [None] * self.total_layers
            for i in range(self.total_layers):
                self.sum_arrs[i] = []
            
            if self.get_coeff == True:
                self.all_coeff_arrs = []  # Calculate all wavelet coefficients, need O(T) storage space
        else:
            raise NotImplementedError
    
    def get_norm(self):
        if self.seq_legth <= 1:
            return 0.0
        
        # Get the full norm
        full_norm = self.total_2_norm
        
        # Calculate the left and right edge of CDJV coeff_arrs
        if self.right_edge is not None and self.right_edge.shape[0] == 3 * self.order - 1:
            self.left_mat = copy.deepcopy(self.left_edge)
            self.right_mat = copy.deepcopy(self.right_edge)
            self.left_mat[:self.order, :] = self.LPREMAT @ self.left_mat[:self.order, :]
            self.right_mat[-self.order:, :] = self.RPREMAT @ self.right_mat[-self.order:, :]
            
            beta = np.concatenate((self.left_mat, self.right_mat), axis=0)
            for i in range(self.total_layers):
                alfa = cdjv_dyad_down(beta, self.HPF, self.LHPEF, self.RHPEF)
                full_norm += np.linalg.norm(alfa) ** 2
                beta = cdjv_dyad_down(beta, self.LPF, self.LLPEF, self.RLPEF)
                if beta.shape[0] < 3 * self.order - 1: break
        return math.sqrt(full_norm)
    
    def get_log2position(self, position):
        # get the update position of each layer,
        # use bitwise operation to speed up.
        update_idx = np.bitwise_and(np.right_shift(position, self.layer_arr), 1)
        update_idx[position < np.left_shift(1, self.layer_arr)] = -1
        return update_idx
    
    def add_signal(self, element):
        # element = element.numpy()
        update_idx_arrs = self.get_log2position(self.seq_legth)
        
        if self.verbose:
            print(self.seq_legth, update_idx_arrs)
        
        none_num = 0
        for i in range(self.total_layers):
            if update_idx_arrs[i] == -1:
                none_num += 1
                update_idx_arrs[i] = 0
            
            if none_num == 2: break
            
            if update_idx_arrs[i] == 0 and (len(self.sum_arrs[i]) == 0 or self.sum_arrs[i][-1][0] is not None):
                self.sum_arrs[i].append([])
                self.sum_arrs[i][-1] = [None] * 2
            
            if i == 0:
                self.sum_arrs[i][-1][update_idx_arrs[i]] = element
            
            if len(self.sum_arrs[i]) > self.order:
                self.sum_arrs[i].pop(0)
            
            if i > 0:
                if update_idx_arrs[i] != -1 and self.sum_arrs[i - 1][-1][1] is not None and len(self.sum_arrs[i - 1]) == self.order:
                    # if self.sum_arrs[i - 1][-1][1] is not None and len(self.sum_arrs[i]) == self.order:
                    # print('Update Sum Arr!')
                    self.sum_arrs[i][-1][update_idx_arrs[i]] = \
                        np.sum(np.array(self.sum_arrs[i - 1]).reshape(-1, self.dim)
                               * self.LPF, axis=0)
            
            if self.verbose:
                print('sum_arrs[{}]:'.format(i), len(self.sum_arrs[i]), self.sum_arrs[i])
            
            if update_idx_arrs[i] == 1 and self.sum_arrs[i][-1][1] is not None and len(self.sum_arrs[i]) == self.order:
                # if self.sum_arrs[i][-1][1] is not None and len(self.sum_arrs[i]) == self.order:
                # Update coeff_arrs
                self.total_2_norm += \
                    np.linalg.norm(np.sum(np.array(self.sum_arrs[i]).reshape(-1, self.dim)
                                          * self.HPF, axis=0)) ** 2  # update 2-norm sequentially
                if self.get_coeff and len(self.sum_arrs[i]) == self.order:
                    # self.all_coeff_arrs.append(self.sum_arrs[i][0] - self.sum_arrs[i][1])
                    self.all_coeff_arrs.append(np.sum(np.array(self.sum_arrs[i]).reshape(-1, self.dim)
                                                      * self.HPF, axis=0))
                if self.verbose:
                    print(i, 'coeff_arrs', self.all_coeff_arrs)
        
        # calculate left edge and right edge.
        if self.left_edge is None:
            self.left_edge = element.reshape(1, self.dim)
        elif self.left_edge.shape[0] < 3 * self.order - 1:
            self.left_edge = np.concatenate((self.left_edge, element.reshape(1, self.dim)), axis=0)
        
        if self.right_edge is None:
            self.right_edge = element.reshape(1, self.dim)
        else:
            self.right_edge = np.concatenate((self.right_edge, element.reshape(1, self.dim)), axis=0)
        if self.right_edge.shape[0] > 3 * self.order - 1:
            self.right_edge = self.right_edge[1:, :]
        
        self.seq_legth += 1
    
    def reinit(self):
        self.seq_legth = 0
        self.total_2_norm = 0.0
        self.coeff_arrs = [None] * (self.total_layers - 1)
        
        self.left_edge = None
        self.right_edge = None
        self.left_mat = None
        self.right_mat = None
        
        self.layer_arr = np.array(range(self.total_layers + 1))
        self.sum_arrs = [None] * self.total_layers
        for i in range(self.total_layers):
            self.sum_arrs[i] = []
        
        if self.get_coeff == True:
            self.all_coeff_arrs = []  # Calculate all wavelet coefficients, need O(T) storage space


# Demo
if __name__ == '__main__':
    # a = StreamingWavelet(10, 10, 2)
    # a.add_signal(np.random.randn(1, 10))
    # a.add_signal(np.random.randn(1, 10))
    # a.add_signal(np.random.randn(1, 10))
    # a.add_signal(np.random.randn(1, 10))
    # a.add_signal(np.random.randn(1, 10))
    # a.add_signal(np.random.randn(1, 10))
    #
    # x = a.get_norm()
    
    # ------------------------------
    # ------------------------------
    
    SW = StreamingWavelet(128, 10000, 1)  # Initialize the Streaming Wavelet Operator
    
    # Generate a sequence of length 10000
    x_list = []
    for i in range(10000):
        x_list.append(np.random.randn(128))
    
    for i in range(10000):
        SW.add_signal(x_list[i])  # Update the wavelet coefficients by adding the new element
        current_norm = SW.get_norm()  # Get the norm of the wavelet coefficients
        print('Norm of Wavelet Coefficients of x_list[0:{}]:'.format(i), current_norm)  # Print the current norm of the wavelet coefficients
