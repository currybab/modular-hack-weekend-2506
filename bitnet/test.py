import torch
from torch.utils import benchmark
from torch import nn

from pack_weight import convert_weight_int8_to_int2
from torch.profiler import profile, record_function, ProfilerActivity
import ctypes
import numpy as np
from max.torch import CustomOpLibrary
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
import torch._dynamo
import random


# torch._dynamo.config.recompile_limit = 256
# set all seed
torch.manual_seed(42)
np.random.seed(42)

bitnet_lib = ctypes.CDLL((Path(__file__).parent / "bitnet_kernels" / "libbitnet.so").__str__())  # Skip CUDA library

def bitnet_int8xint2_linear_cuda(input0, input1, s, ws, ret):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])

    return ret

def bitnet_int8xint2_linear_mojo(input0, input1, s, ws, ret):
    # Load our custom operations
    mojo_kernels = Path(__file__).parent / "bitnet_kernels"
    ops = CustomOpLibrary(mojo_kernels)

    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]
    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4
    bitlinear = ops.bitlinear[{"M": M, "N": N, "K": K, "dtype": DType.bfloat16}]
    torch.compile(bitlinear)(ret, input0, input1, s, ws) 
    # bitlinear(ret, input0, input1, s, ws)

    return ret

def bitnet_int8xint2_linear_max(input0, input1, s, ws, ret):
    # Load our custom operations
    mojo_kernels = Path(__file__).parent / "bitnet_kernels"

    # Define device (use GPU if available) - create fresh session each time
    device = Accelerator() if accelerator_count() > 0 else CPU()
    session = InferenceSession(devices=[device])

    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]
    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    # Convert torch tensors to numpy, then to MAX AI tensors
    # Note: BFloat16 is not supported by numpy, so convert to float32 first
    input0_numpy = input0.detach().cpu().numpy()
    input1_numpy = input1.detach().cpu().numpy()
    s_numpy = s.detach().cpu().float().numpy()  # Convert BFloat16 to float32
    ws_numpy = ws.detach().cpu().float().numpy()  # Convert BFloat16 to float32
    
    input0_tensor = Tensor.from_numpy(input0_numpy).to(device)
    input1_tensor = Tensor.from_numpy(input1_numpy).to(device)
    s_tensor = Tensor.from_numpy(s_numpy).to(device)
    ws_tensor = Tensor.from_numpy(ws_numpy).to(device)

    with Graph(
        "bit_linear_graph",
        input_types=[
            TensorType(
                DType.int8,
                shape=input0_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.int8,
                shape=input1_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,  # Changed from bfloat16 to float32
                shape=s_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,  # Changed from bfloat16 to float32
                shape=ws_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Define inputs to the graph
        input0_val, input1_val, s_val, ws_val = graph.inputs
    
        # custom op call
        output = ops.custom(
            name="bitlinear",
            values=[input0_val, input1_val, s_val, ws_val],
            device=DeviceRef.from_device(device),
            out_types=[
                TensorType(
                    dtype=DType.float32,  # Match input dtype
                    shape=out_shape,
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={
                "M": M,
                "N": N, 
                "K": K,
                "dtype": DType.float32,
            },
        )[0].tensor
        
        graph.output(output)

    # Compile the graph
    print("Compiling bit linear graph...")
    try:
        model = session.load(graph)
        print("Graph compiled successfully!")
    except Exception as e:
        print(f"Compilation failed: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

    # Execute the operation
    print("Executing bit linear...")
    result = model.execute(input0_tensor, input1_tensor, s_tensor, ws_tensor)[0]

    # Copy values back to the CPU and convert to torch tensor
    assert isinstance(result, Tensor)
    # result_cpu = result.to(CPU())
    result_numpy = result.to_numpy()  # Use to_numpy() instead of numpy()
    
    # Make a writable copy to avoid PyTorch warning
    result_numpy_writable = result_numpy.copy()
    
    # Convert back to torch tensor and move to same device as input
    result_torch = torch.from_numpy(result_numpy_writable).to(ret.device)
    ret.copy_(result_torch)
    
    return ret


bitnet_int8xint2_linear = bitnet_int8xint2_linear_mojo

if __name__ == '__main__':
    test_list = [
        (2560,  2560), 
        (3840,  2560), 
        (2560,  6912) ,
        (3200, 3200), 
        (4800, 3200), 
        # (3200, 10240),
        # (13824, 2560),
        # (20480, 3200),
    ]

    N,K = test_list[random.randint(0, len(test_list) - 1)]
    print(f"Shape: {N}x{K}")
    # Clean up torch/CUDA state before each test
    # torch.cuda.empty_cache()  # Clear GPU memory cache
    # torch.cuda.synchronize()  # Synchronize all CUDA operations
    # torch.cuda.reset_peak_memory_stats()  # Reset memory statistics
        
    # # Clear torch compile cache to avoid conflicts
    # torch._dynamo.reset()  # Reset dynamo compilation cache
        
    # For even more thorough cleanup, you can uncomment this:
    # import gc
    # gc.collect()  # Python garbage collection
    # torch.cuda.ipc_collect()  # CUDA IPC cleanup
        
    weight = torch.randint(-1, 2, (N, K), dtype=torch.int8, device='cuda')
    weight_scale = torch.ones(1, dtype=torch.bfloat16, device='cuda')
    weight_compressed = convert_weight_int8_to_int2(weight).to('cuda')

    input0 = torch.randint(-128,127,(1, K),dtype=torch.int8, device='cuda')
    input0_bf16 = input0.to(torch.bfloat16)
    input_np = input0.cpu().to(torch.int32).numpy()
    weight_np = weight.cpu().to(torch.int32).T.numpy()
    out_np = np.matmul(input_np,weight_np)
    out_np = torch.tensor(out_np).cuda().to(torch.bfloat16)

    s = torch.ones(1, dtype=torch.bfloat16, device='cuda')
    ws = torch.ones(6, dtype=torch.bfloat16, device='cuda')

    # Test both CUDA and Mojo kernels
    ret_mojo = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
    ret_cuda = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
    # ret_max = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
    
    # Run only one Mojo implementation to avoid conflicts
    out_mojo = bitnet_int8xint2_linear_mojo(input0, weight_compressed, s, ws, ret_mojo)
    out_cuda = bitnet_int8xint2_linear_cuda(input0, weight_compressed, s, ws, ret_cuda)
    # out_max = bitnet_int8xint2_linear_max(input0, weight_compressed, s, ws, ret_max)

    print(f'Mojo == NumPy: {torch.allclose(ret_mojo, out_np, atol=1e-3)}')
    print(f'CUDA == NumPy: {torch.allclose(ret_cuda, out_np, atol=1e-3)}')
    # print(f'Max == NumPy: {torch.allclose(ret_max, out_np, atol=1e-3)}')

    # input0 = torch.randint(-128,127,(1, K),dtype=torch.int8, device='cuda')
    # input0_fp16 = input0.to(torch.float16)
    # input0_bf16 = input0.to(torch.bfloat16)
    # weight_fp16 = weight.to(torch.float16).T
    # weight_bf16 = weight.to(torch.bfloat16).T
    # ret = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
    # s = torch.ones(1, dtype=torch.bfloat16, device='cuda')
    # ws = torch.ones(6, dtype=torch.bfloat16, device='cuda')
    # t0 = benchmark.Timer(
    #     stmt="bitnet_int8xint2_linear(input0, weight_compressed, s, ws, ret)",
    #     setup="from __main__ import input0, weight_compressed, s, ws, ret, bitnet_int8xint2_linear",
    #     num_threads=1,
    # )

    # t1 = benchmark.Timer(
    #     stmt="torch.matmul(input0_bf16,weight_bf16)",
    #     setup="from __main__ import input0_bf16, weight_bf16",
    #     num_threads=1,
    # )

    # time0 = t0.timeit(50)
    # time1 = t1.timeit(50)

    # print(f'Shape{N,K}, W2A8: {time0.mean * 1e6:.2f}us, torch BF16: {time1.mean * 1e6:.2f}us')

    
