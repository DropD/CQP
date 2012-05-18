import numpy as np

class numerov(object):
    __slots__=['dx', 'dx_fac']

    def __init__(self, dx):
        self.dx = dx

    def __setattr__(self, name, value):
        if name == 'dx':
            object.__setattr__(self, 'dx', value)
            object.__setattr__(self, 'dx_fac', value * value / 12)
        elif name == 'dx_fac':
            raise Exception('numerov: dx_fac should not be set manually')

    def step(self, x, k, psi):
        a = (2 - (k(x + self.dx) * 10 * self.dx_fac)) * psi[1]
        b = (1 + (k(x)                * self.dx_fac)) * psi[0]
        c = (1 + (k(x + 2 * self.dx)  * self.dx_fac))
        return (a - b) / c

    #def integrate(self, x_begin, x_end, k, psi):
    #    psi_nm1 = psi[0]
    #    psi_n   = psi[1]
    #    n = int((x_end - x_begin) / self.dx)
    #    for i in range(0, n):
    #        xn = x_begin + i * self.dx
    #        psi_np1 = step(xn, k, psi)
    #        psi_nm1 = psi_n
    #        psi_n = psi_np1
    #    return [psi_nm1, psi_n]
    
    def integrate(self, x_begin, x_end, k, psi0):
        n = int((x_end - x_begin) / self.dx)
        psi = np.zeros(n, dtype = np.complex)
        psi[0] = psi0[0]
        psi[1] = psi0[1]
        for i in range(0, n-2):
            xn = x_begin + i * self.dx
            psi[i+2] = self.step(xn, k, psi[i:i+2])
        return psi
