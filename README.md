# MicroGPT C Implementation

A highly optimized C implementation of Andrej Karpathy's microgpt.py, demonstrating significant performance improvements over the original Python version while maintaining identical algorithmic behavior.

## Overview

This implementation is a complete, dependency-free C port of the microGPT language model that trains on character-level text data. It implements a minimal transformer architecture with multi-head self-attention, RMS normalization, and Adam optimization.

## Features

- **Zero dependencies**: Pure C implementation using only standard library
- **Identical algorithm**: Exact match to Python original's behavior
- **Extreme performance**: 700+× speedup over Python implementation
- **Memory efficient**: Manual memory management with cache-friendly data layouts
- **Self-contained**: Automatically downloads training data on first run

## Model Architecture

```c
#define N_LAYER     1        // Number of transformer layers
#define N_EMBD      16       // Embedding dimension
#define N_HEAD      4        // Number of attention heads
#define HEAD_DIM    4        // Dimension per head (N_EMBD / N_HEAD)
#define FF_DIM      64       // Feed-forward dimension (4 * N_EMBD)
#define BLOCK_SIZE  16       // Maximum sequence length
```

## Performance Benchmarks

Based on tests with 1000 training steps on the names dataset:

| Metric | Python | C | Speedup |
|--------|--------|---|---------|
| **Training Time** | 272.23s | 0.37s | **736×** |
| **Inference Time** | 2.33s | 0.01s | **233×** |
| **Total Time** | 274.57s | 0.39s | **704×** |
| **Time per Step** | 272.23ms | 0.37ms | **736×** |
| **Time per Sample** | 116.68ms | 0.60ms | **194×** |

### Model Specifications
- **Parameters**: 4,192 trainable parameters
- **Vocabulary Size**: 27 characters (a-z + BOS token)
- **Training Dataset**: 32,033 names
- **Training Steps**: 1,000 iterations

## Compilation & Usage

### Prerequisites
- GCC compiler (or compatible C compiler)
- curl (for automatic dataset download)

### Compilation
```bash
gcc -O3 -march=native -ffast-math -o gpt microgpt.c -lm
```

### Running
```bash
./gpt
```

On first run, the program automatically downloads the names.txt dataset from Karpathy's makemore repository.

## Technical Implementation Details

### Performance Optimizations

1. **Flat Memory Layout**: All tensors stored as contiguous float arrays
2. **Manual Backpropagation**: Eliminates computational graph overhead
3. **Cache-Friendly Loops**: Optimized memory access patterns
4. **Inline Math Primitives**: Custom matrix operations with restrict pointers
5. **Static Allocation**: Fixed-size buffers avoid dynamic allocation overhead

### Key Components

- **Tokenizer**: Character-level vocabulary with BOS token
- **Transformer Layers**: Multi-head attention with causal masking
- **RMS Normalization**: Root Mean Square normalization
- **Adam Optimizer**: With linear learning rate decay
- **Inference**: Autoregressive generation with temperature sampling

### Memory Management

The implementation uses a global tensor registry system:
- All parameters registered for optimizer access
- Separate forward/backward activation buffers
- Adam momentum and variance buffers
- Manual cleanup not required (program lifetime)

## Training Output Example

```
num docs:   32033
vocab size: 27
num params: 4192

Starting training for 1000 steps...
step 1000 / 1000 | loss 2.8979
Training completed in 0.37 seconds
Average time per step: 0.37 ms

--- inference (new, hallucinated names) ---
sample  1: jarile
sample  2: alana
sample  3: lenanh
sample  4: jara
sample  5: keleli
...
sample 20: jaish

Inference completed in 0.01 seconds
Average time per sample: 0.60 ms
```

## Hyperparameters

```c
// Training
#define NUM_STEPS   1000
#define LR          0.01f
#define BETA1       0.85f
#define BETA2       0.99f
#define ADAM_EPS    1e-8f

// Model
#define INIT_STD    0.08f
#define TEMPERATURE 0.5f
#define N_SAMPLES   20
```

## Comparison with Python Original

The C implementation achieves remarkable performance gains through:

1. **Elimination of Python interpreter overhead**
2. **Direct memory access without object indirection**
3. **Optimized linear algebra primitives**
4. **Reduced function call overhead**
5. **Better cache utilization with contiguous memory**

Despite these optimizations, the algorithm remains identical to the Python version, producing equivalent training loss and generation quality.



## Acknowledgments

- Original Python implementation by Andrej Karpathy
- Dataset from makemore repository (names.txt)
- Inspiration from nanoGPT and related projects

## License

This implementation follows the same license as the original microgpt.py project.

