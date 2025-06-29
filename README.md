# "Weekend Experiment🧪: CUDA Kernel → Mojo 🔥"

> **Spoiler alert**: Got it working for some cases, but hit some walls too (probably because I'm a GPU newbie trying to learn Mojo at the same time) 😅

## 🤔 What's this about?

I was browsing GitHub and stumbled upon [Microsoft's BitNet repo](https://github.com/microsoft/BitNet). Cool 1-bit neural networks, right? But then I noticed they had this hand-optimized CUDA kernel for `bitlinear_int8xint2` operations. 

Being the curious developer I am, I thought...

**"Hmm, I wonder if I could rewrite just this CUDA kernel in Mojo?"** 

So I grabbed that specific kernel and spent my weekend trying to make Mojo do the same thing.

*(Okay, full confession: I initially thought "How hard can it be? Maybe I can rewrite the whole thing!" But after diving in... let's just say I got humbled real quick 😅)*

Turns out, just this one gnarly GPU kernel was plenty challenging!

**The target**: [This specific CUDA kernel](https://github.com/microsoft/BitNet/tree/main/gpu/bitnet_kernels) (about 120 lines of highly optimized CUDA)

## 🎉 Plot twist: It kinda worked! (with some caveats)

**The good news:**
- ✅ **Core logic works**: Successfully ported the CUDA kernel structure to Mojo
- ✅ **Performance is competitive**: Timing shows Mojo can hang with CUDA 💪
- ✅ **Correctness on happy path**: Results match perfectly for many test cases

**The reality check:**
- ⚠️ **Stability issues**: Memory crashes on large arrays and sequential runs
- ⚠️ **Edge cases bite**: Some matrix configurations cause `CUDA_ERROR_ILLEGAL_ADDRESS`
- 🤔 **Integration challenges**: Single kernel calls work, but model integration is tricky

**Bottom line**: Promising proof-of-concept, but needs more work for production use!

## 🤓 What I learned (the hard way)

1. **Mojo can compete with CUDA** - When it works, performance is impressive! 🤯
2. **Memory management is HARD** - GPU crashes taught me about proper synchronization
3. **CUDA intrinsics are magical** - Reverse-engineering `__dp4a` and friends into Mojo SIMD was like solving puzzles blindfolded
4. **Bit manipulation is an art** - Those packed int8×int2 formats are sneaky
5. **Double learning curve is real** - Learning GPU programming AND Mojo simultaneously was... ambitious 😅

## 😎 How I pulled this off

### The main characters

**`bitnet.mojo`** - The star of the show
- Rewrote all that CUDA wizardry using Mojo SIMD ops
- Added timing because I'm obsessed with benchmarks
- Supports different matrix sizes (because one size fits none)

**`bitnet_kernels.cu`** - The CUDA veteran
- The original badass kernel I'm trying to beat
- Also added timing to see who's faster 🏁

**`test.py`** - The judge and jury
- Tests if my Mojo code actually works (spoiler: it does!)
- Benchmarks everything because numbers are fun
- Multiple ways to call Mojo because I couldn't decide

### The cool technical stuff

- **Bit manipulation mastery**: Handled those tricky int8×int2 packed formats
- **GPU memory gymnastics**: Made sure everything loads efficiently
- **Warp-level programming**: Mojo supports warp reductions and shuffle operations like CUDA!
- **PyTorch friendship**: Got Mojo and PyTorch to work together nicely

## 📊 Performance Results

```
$ pixi run python bitnet/test.py
Shape: 4800x3200
Mojo kernel execution time: 12.236495997058228 ms
CUDA kernel execution time: 7.12717 ms
Mojo == NumPy: True
CUDA == NumPy: True

$ pixi run python bitnet/test.py
Shape: 2560x6912
Mojo kernel execution time: 7.668589008972049 ms
CUDA kernel execution time: 6.30867 ms
Mojo == NumPy: True
CUDA == NumPy: True

$ pixi run python bitnet/test.py
Shape: 3840x2560
Mojo kernel execution time: 7.756931008771062 ms
CUDA kernel execution time: 6.02979 ms
Mojo == NumPy: True
CUDA == NumPy: True

$ pixi run python bitnet/test.py
Shape: 3200x3200
Mojo kernel execution time: 6.379697006195784 ms
CUDA kernel execution time: 6.35187 ms
Mojo == NumPy: True
CUDA == NumPy: True
```

**TL;DR: Mojo works... when it works! 😅**
- Mojo vs CUDA: Usually within 6ms on single runs (not bad!)
- Correctness: 100% on supported matrix sizes
- Reality check: Only tested on individual kernel calls, not real workloads
- **The gotcha**: Memory issues on large arrays and multi-run scenarios

## 🛠️ How to Run

### Prerequisites
```bash
# Install dependencies
pixi install
cd bitnet/bitnet_kernels
bash compile.sh
```

### Running Tests
```bash
# Run performance and correctness tests
pixi run python bitnet/test.py
```

## 🏗️ Project Structure

```
bitnet/
├── bitnet_kernels/
│   ├── bitnet.mojo              # Main Mojo implementation
│   ├── bitnet_kernels.cu        # CUDA reference implementation
│   └── bitnet_kernels.h         # CUDA kernel headers
├── dot4_test/                   # Debug tests for dot product operations
│   ├── simple_mojo_test.mojo    # Mojo SIMD dot product tests
│   └── test_dot4.cu             # CUDA __dp4a comparison tests
├── decode_test/                 # Debug tests for int2 decoding
│   ├── test_mojo_decode.mojo    # Mojo decode implementation tests
│   └── test_decode.cu           # CUDA decode comparison tests
└── test.py                      # Main testing and benchmarking framework
```

## 🔮 What's next? (if I had infinite time)

- [ ] **Fix the memory issues** - Figure out why large arrays crash
- [ ] **Solve sequential execution** - Stop the `CUDA_ERROR_ILLEGAL_ADDRESS` drama
- [ ] **Handle batch sizes > 1** - The CUDA kernel supports it, mine doesn't yet
- [ ] **Actually integrate into BitNet models** - The real test of usefulness

## 🙏 Acknowledgments

Built during **Modular Hack Weekend** using:
- [Modular MAX AI](https://modular.com/max) for Mojo implementation
- [Microsoft BitNet](https://github.com/microsoft/BitNet) for reference CUDA kernels
