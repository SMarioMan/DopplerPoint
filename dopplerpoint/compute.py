"""Optional CUDA spectral interpolation; geometry/filtering stays on the CPU."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class SpectralSampler:
    def __init__(self, t, f, values, backend='cpu'):
        self.backend = backend
        if backend == 'cpu':
            self.sampler = RegularGridInterpolator((t,f),values,bounds_error=False,fill_value=-5.)
        elif backend == 'cuda':
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise ValueError('No CUDA device available')
            except ImportError as e:
                raise ValueError('CUDA backend requires a matching CuPy installation (e.g. cupy-cuda12x)') from e
            self.cp = cp
            self.t, self.f, self.values = cp.asarray(t),cp.asarray(f),cp.asarray(values)
        else:
            raise ValueError('backend must be cpu or cuda')

    def __call__(self, points):
        if self.backend == 'cpu':
            return self.sampler(points)
        cp = self.cp
        p = cp.asarray(points)
        x,y = p[...,0],p[...,1]
        i = cp.clip(cp.searchsorted(self.t,x)-1,0,len(self.t)-2)
        j = cp.clip(cp.searchsorted(self.f,y)-1,0,len(self.f)-2)
        a = (x-self.t[i])/(self.t[i+1]-self.t[i])
        b = (y-self.f[j])/(self.f[j+1]-self.f[j])
        v = ((1-a)*(1-b)*self.values[i,j]+a*(1-b)*self.values[i+1,j]
             +(1-a)*b*self.values[i,j+1]+a*b*self.values[i+1,j+1])
        valid = (x>=self.t[0])&(x<=self.t[-1])&(y>=self.f[0])&(y<=self.f[-1])
        return cp.asnumpy(cp.where(valid,v,-5.))
