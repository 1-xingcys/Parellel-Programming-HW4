# Parallel Programming HW4 - CUDA Bitcoin Miner

## Project Overview

This project implements a CUDA-parallelized Bitcoin mining program that uses GPU acceleration to search for a nonce value satisfying the target difficulty.

## Implementation

### Parallelization

- The original sequential for-loop is distributed across multiple GPU threads for parallel processing
- Each thread tests a different nonce value (`global_tid`, `global_tid + thread_count`, ...)
- A flag in global memory marks whether the target nonce has been found
- Once any thread finds the answer, all threads can detect it immediately and terminate early

### Memory Optimization

1. **Constant Memory**

   - Uses `cudaMemcpyToSymbol` to copy the block template and target difficulty into constant memory
   - Provides efficient read-only access

2. **SHA-256 Optimization**

   - Specializes the SHA-256 function for fixed lengths (80 bytes and 32 bytes), removing unnecessary length checks
   - Replaces the original 64-word array with a 16-word circular buffer so it can fit into registers
   - Unrolls loops and uses scalar variables to eliminate array index computation

3. **Midstate Precomputation**
   - Precomputes the hash of the invariant part of the block header (midstate) on the CPU
   - Each thread continues computation from the midstate, skipping one `sha256_transform` call
   - Significantly improves overall performance

## Build and Run

### Build

```bash
cd b10705009
make
```

### Run

```bash
./hw4 <input_file> <output_file>
```

Example:

```bash
./hw4 ../testcases/case00.in output.out
```

### Clean

```bash
make clean
```

## File Structure

```
├── hw4.cu           # Main program (CUDA kernel and host code)
├── sha256.cu        # SHA-256 implementation (CPU and GPU versions)
├── sha256.h         # SHA-256 header file
├── serial.cu        # Non-parallelized sequential version
├── Makefile         # Build configuration
└── report.pdf       # Experiment report
```
