# FPGA Matrix Multiplication — Vitis HLS & Alveo U200

Hardware-accelerated matrix multiplication implemented using **AMD/Xilinx Vitis HLS** and executed on an **AMD/Xilinx Alveo U200 FPGA accelerator card**.

The project explores how a matrix multiplication algorithm can be mapped from a conventional CPU implementation to custom FPGA hardware using **High-Level Synthesis (HLS)**, with loop pipelining and array partitioning used to expose parallelism.

---

## Overview

Matrix multiplication is a fundamental operation in scientific computing, computer vision, machine learning, and many other computational workloads.

For matrices

$$
A \in \mathbb{Z}^{n \times m}
$$

and

$$
B \in \mathbb{Z}^{m \times p}
$$

the resulting matrix

$$
C = A \times B
$$

has dimensions

$$
C \in \mathbb{Z}^{n \times p}
$$

with each element computed as

$$
C_{ij} = \sum_{k=0}^{m-1} A_{ik}B_{kj}
$$

This project implements this computation as an FPGA kernel using **Vitis HLS**.

The host application:

1. Generates input matrices.
2. Loads the compiled FPGA binary (`.xclbin`).
3. Programs the Alveo U200.
4. Allocates FPGA global-memory buffers.
5. Transfers input data to the FPGA.
6. Launches the hardware kernel.
7. Transfers the result back to the host.
8. Compares the FPGA result with a CPU reference implementation.
9. Reports execution timing.

---

## Hardware

### Target FPGA

**AMD/Xilinx Alveo U200**

The Alveo U200 is a data-center accelerator card based on an FPGA architecture designed for high-throughput compute workloads.

The project uses the FPGA as a custom matrix-multiplication accelerator rather than executing the computation directly on the host CPU.

---

## Software

The project uses:

* **AMD/Xilinx Vitis**
* **Vitis HLS**
* **OpenCL**
* **C/C++**
* `xcl2.hpp`
* Custom `EventTimer` utility
* FPGA `.xclbin` binary

---

## Project Architecture

The overall execution flow is:

```text
                 Host CPU
                    |
                    |
          +---------v---------+
          | Generate matrices |
          |      A and B      |
          +---------+---------+
                    |
                    | OpenCL
                    v
          +-------------------+
          |   Alveo U200      |
          |                   |
          |  +-------------+  |
          |  | Matrix      |  |
          |  | Multiply    |  |
          |  | HLS Kernel  |  |
          |  +-------------+  |
          |                   |
          +---------+---------+
                    |
                    | Result C
                    v
          +-------------------+
          |     Host CPU      |
          |                   |
          | Compare against   |
          | CPU reference     |
          +-------------------+
```

---

## Matrix Dimensions

The current implementation defines:

```cpp
#define lm 4
#define ln 4
#define lp 4

#define m (1 << lm)
#define n (1 << ln)
#define p (1 << lp)
```

Therefore:

```text
n = 16
m = 16
p = 16
```

The matrices are consequently:

```text
A : 16 × 16
B : 16 × 16
C : 16 × 16
```

The matrix multiplication is:

```text
C[16][16] = A[16][16] × B[16][16]
```

---

# FPGA Kernel

The main HLS implementation is:

```cpp
void Matrix_Mult(int A[n][m],
                 int B[m][p],
                 int C[n][p])
{
    #pragma HLS ARRAY_PARTITION variable=A complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C complete dim=2

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {

            #pragma HLS PIPELINE II=1

            for(int k = 0; k < p; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
```

---

## HLS Optimizations

Two main HLS techniques are used.

### 1. Array Partitioning

The kernel contains:

```cpp
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=C complete dim=2
```

Array partitioning changes how arrays are represented in FPGA hardware.

Instead of treating an array as one monolithic memory structure, Vitis HLS can split it into multiple independent storage elements.

This can increase the number of memory accesses that can occur in parallel.

For example:

```cpp
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
```

partitions the second dimension of `A`.

Similarly:

```cpp
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
```

partitions the first dimension of `B`.

This is useful for matrix multiplication because the kernel repeatedly accesses different elements of rows and columns.

---

### 2. Loop Pipelining

The inner computation is pipelined using:

```cpp
#pragma HLS PIPELINE II=1
```

The **Initiation Interval (II)** specifies how frequently a new iteration can begin.

An II of:

```text
II = 1
```

means the HLS scheduler attempts to start a new loop iteration every clock cycle.

This allows operations from multiple iterations to overlap in hardware.

Conceptually:

```text
Cycle:    1   2   3   4   5   6
Iter 1:  [====]
Iter 2:      [====]
Iter 3:          [====]
Iter 4:              [====]
```

rather than executing every iteration completely before starting the next one.

---

# Host Application

The host application uses the OpenCL API to communicate with the FPGA.

The main steps are:

### 1. Generate Input Data

The host creates matrices and initializes them with random values:

```cpp
std::vector<int32_t> A(n * m);
std::vector<int32_t> B(m * p);
std::vector<int32_t> D(n * p);

for(int i = 0; i < n*m; i++)
    A[i] = rand() % 256;

for(int i = 0; i < m*p; i++)
    B[i] = rand() % 256;
```

---

### 2. Load the FPGA Binary

The executable expects the compiled FPGA binary as its command-line argument:

```bash
./host <XCLBIN_FILE>
```

The host loads the `.xclbin` file using:

```cpp
auto fileBuf = xcl::read_binary_file(binaryFile);
```

The binary is then used to program the FPGA.

---

### 3. Detect the FPGA

The Xilinx utility:

```cpp
xcl::get_xil_devices();
```

is used to locate available Xilinx/AMD FPGA devices.

The host attempts to program each available device until a compatible device is found.

---

### 4. Create OpenCL Buffers

The host allocates buffers for communication between CPU memory and FPGA global memory.

The buffers use:

```cpp
CL_MEM_USE_HOST_PTR
```

to allow the OpenCL runtime to use the supplied host memory where possible.

The project also uses:

```cpp
aligned_allocator<int>
```

for appropriately aligned host allocations.

---

### 5. Transfer Input Data

Input buffers are migrated to the FPGA:

```cpp
q.enqueueMigrateMemObjects(
    {buffer_in1, buffer_in2},
    0
);
```

The direction `0` indicates a host-to-device transfer.

---

### 6. Launch the FPGA Kernel

The kernel is launched using:

```cpp
q.enqueueTask(krnl_vector_add);
```

The project is based on the original Vitis vector-addition example, and this section is currently being adapted for the matrix-multiplication kernel.

---

### 7. Transfer Results Back

After the FPGA completes the computation:

```cpp
q.enqueueMigrateMemObjects(
    {buffer_output},
    CL_MIGRATE_MEM_OBJECT_HOST
);
```

moves the result back to host memory.

---

### 8. Validate the Result

The host compares the FPGA result against a software reference:

```cpp
if (source_hw_results[i] != source_sw_results[i]) {
    std::cout << "Error: Result mismatch" << std::endl;
    match = false;
}
```

The program reports:

```text
TEST PASSED
```

when the results match.

---

# Performance Measurement

The host uses the `EventTimer` utility to measure different stages of execution.

Examples include:

```text
Allocate Memory in Host Memory
Fill the buffers
Load Binary File to Alveo U200
Allocate Buffer in Global Memory
Set the Kernel Arguments
Copy input data to device global memory
Launch the Kernel
Copy Result from Device Global Memory to Host Local Memory
Compare the results
```

The final timing report is generated using:

```cpp
et.print();
```

This allows the execution pipeline to be analyzed beyond just the kernel computation.

---

# Compilation Flow

The project follows the typical Vitis acceleration workflow:

```text
C/C++ HLS Kernel
       |
       v
   Vitis HLS
       |
       v
 RTL / FPGA implementation
       |
       v
    .xclbin
       |
       v
Host OpenCL application
       |
       v
    Alveo U200
```

The HLS kernel is synthesized and compiled into an FPGA binary.

The host application subsequently loads this binary at runtime.

---

# Expected Execution

The host application is designed to be executed with:

```bash
./host <matrix_multiplication.xclbin>
```

For example:

```bash
./host matrix_mult.xclbin
```

The application should:

1. Detect the Alveo U200.
2. Load the `.xclbin`.
3. Program the FPGA.
4. Allocate the required buffers.
5. Transfer input data.
6. Execute the matrix multiplication kernel.
7. Transfer the result back.
8. Validate the result.
9. Print execution timings.

---

# Matrix Multiplication Algorithm

The mathematical operation performed by the kernel is:

```text
for i = 0 ... n-1
    for j = 0 ... p-1
        for k = 0 ... m-1
            C[i][j] += A[i][k] × B[k][j]
```

For the current configuration:

```text
n = m = p = 16
```

so the kernel performs:

```text
16 × 16 × 16 = 4096
```

multiply-accumulate iterations.

The computational complexity is:

$$
O(nmp)
$$

and for square matrices:

$$
O(N^3)
$$

---

# FPGA Parallelism

The primary motivation for implementing the algorithm on an FPGA is to exploit hardware-level parallelism.

The CPU version executes the algorithm through sequential instructions, whereas HLS can transform the C/C++ description into dedicated hardware.

The design uses:

```text
Array Partitioning
        +
Loop Pipelining
        +
Dedicated FPGA arithmetic
```

to expose parallelism.

A simplified conceptual representation is:

```text
             A values
                |
        +-------+-------+
        |               |
        v               v
      A[i][0]         A[i][1]
        |               |
        v               v
     +------+        +------+
     | MUL  |        | MUL  |
     +--+---+        +--+---+
        |               |
        +-------+-------+
                |
                v
              ADD
                |
                v
             C[i][j]
```

The actual hardware architecture is determined by the Vitis HLS scheduler and synthesis configuration.

---
