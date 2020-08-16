import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

from pycuda.elementwise import ElementwiseKernel

if __name__ == "__main__":
    dim = 1024
    x = np.linspace(-1, 1, dim, dtype=np.float32)
    y = np.linspace(-1, 1, dim, dtype=np.float32)
    out = np.empty((dim*dim), dtype=np.float32)

    x_gpu = cuda.mem_alloc(x.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)
    out_gpu = cuda.mem_alloc(out.nbytes)


    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(y_gpu, y)
    cuda.memcpy_htod(out_gpu, out)

    with open("kernels.cuda", "r") as kernel_file:
        source_module = SourceModule(kernel_file.read())

    hypot = source_module.get_function("hypotenuse")
    hypot(x_gpu, y_gpu, out_gpu, block=(dim,1,1), grid=(dim,1,1))

    cuda.memcpy_dtoh(out, out_gpu)

    out = out.reshape((dim,dim))
    out = out


