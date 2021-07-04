import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

with open("kernels.cu", 'r') as f:
    source_module = SourceModule(f.read())
# advect = source_module.get_function("advect")
diffuse = source_module.get_function("diffuse")
fieldTex = source_module.get_texref("fieldTex")

width = 200
initial_val = np.zeros((width,width))
initial_val[width//4:3*width//4, width//4:3*width//4] = 1

# field1 = cuda.Array(size=(width,width), dtype=np.float32)
field1 = cuda.mem_alloc(initial_val.size)
fieldTex.set_address_2d(field1)
fieldTex.set_array(field1)
# cuda.bind_array_to_texref(field1, fieldTex)


# cuda.memcpy_htod(fieldTex, initial_val.astype(np.float32))
result = diffuse()


