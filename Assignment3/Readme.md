# FPGA Matrix Multiplication with 512-bit Vectorization

Hardware-accelerated matrix computation using **AMD/Xilinx Vitis HLS**, **OpenCL**, and an **Alveo U200 FPGA**.

This project investigates FPGA acceleration techniques for matrix multiplication using:

* 512-bit wide data types
* AXI memory interfaces
* Multiple global-memory bundles
* Local FPGA buffering
* HLS pipelining
* HLS loop unrolling
* HLS dataflow
* CPU reference computation
* Host-to-FPGA and FPGA-to-host data transfers

The design explores how increasing memory width and exposing parallel computation can improve the throughput of a matrix-based workload.

---

# Overview

The project implements a hardware kernel named `vadd` that operates on matrices represented using a **512-bit integer datatype**:

```cpp
typedef ap_uint<DATAWIDTH> uint512_dt;
```

with:

```cpp
#define DATAWIDTH 512
#define VECTOR_SIZE (DATAWIDTH / 32)
```

Therefore:

```text
512 bits / 32 bits = 16 × 32-bit values
```

can be packed into a single `uint512_dt`.

The FPGA kernel processes matrix data using these wide words and uses HLS directives to expose parallelism inside the computation.

The host application uses OpenCL to:

1. Generate input data.
2. Calculate a software reference result.
3. Load the FPGA `.xclbin`.
4. Program the Alveo U200.
5. Allocate global-memory buffers.
6. Transfer input data to the FPGA.
7. Launch the hardware kernel.
8. Retrieve the result.
9. Compare FPGA and CPU results.
10. Report execution times.

---

# Hardware

## Target Platform

**AMD/Xilinx Alveo U200**

The Alveo U200 is used as the hardware accelerator while the host CPU manages:

* Input generation
* FPGA programming
* Memory transfers
* Kernel configuration
* Result validation
* Performance measurement

---

# Software Stack

| Technology               | Purpose                            |
| ------------------------ | ---------------------------------- |
| C/C++                    | Host and kernel implementation     |
| Vitis HLS                | C/C++ → FPGA hardware synthesis    |
| OpenCL                   | Host/FPGA communication            |
| Xilinx Runtime utilities | Device and `.xclbin` management    |
| `ap_int.h`               | Arbitrary-width FPGA integer types |
| `hls_stream.h`           | HLS streaming support              |
| `xcl2.hpp`               | Xilinx OpenCL helper functions     |
| `EventTimer`             | Execution-time measurement         |
| Alveo U200               | Target FPGA accelerator            |

---

# Design Parameters

The kernel defines:

```cpp
#define lm 6
#define ln 6
#define lp 6

#define m (1 << lm)
#define n (1 << ln)
#define p (1 << lp)
```

which gives:

```text
M = 64
N = 64
P = 64
```

The intended matrix dimensions are therefore:

```text
A : 64 × 64
B : 64 × 64
C : 64 × 64
```

The kernel also defines:

```cpp
#define BUFFER_SIZE 64
#define DATAWIDTH 512
#define VECTOR_SIZE (DATAWIDTH / 32)
```

Therefore:

```text
BUFFER_SIZE = 64
DATAWIDTH   = 512 bits
VECTOR_SIZE = 16 × 32-bit elements
```

---

# 512-bit Vectorization

One of the main features of this implementation is the use of:

```cpp
typedef ap_uint<512> uint512_dt;
```

instead of a conventional 32-bit integer.

A 512-bit word can contain:

```text
16 × 32-bit integers
```

Conceptually:

```text
512-bit word
+---------+---------+---------+-----+---------+
| int[0]  | int[1]  | int[2]  | ... | int[15] |
+---------+---------+---------+-----+---------+
   32b       32b       32b              32b
```

The kernel extracts individual 32-bit elements using:

```cpp
tmp1 = v1_local[k].range(
    32 * (k + 1) - 1,
    k * 32
);
```

and similarly for `tmp2`.

This allows the design to move and process much wider memory words than a conventional 32-bit interface.

---

# FPGA Kernel

The kernel is declared as:

```cpp
extern "C" {
    void vadd(
        const uint512_dt *in1,
        const uint512_dt *in2,
        uint512_dt *out,
        int N,
        int M,
        int P
    )
}
```

Despite the kernel retaining the name `vadd` from the original Vitis example, the computation has been modified to perform matrix-oriented operations.

---

# AXI Memory Interfaces

The kernel uses three separate AXI master interfaces:

```cpp
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem
#pragma HLS INTERFACE m_axi port = in2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out bundle = gmem2
```

This creates separate memory interfaces for:

```text
             +-------------+
             | FPGA Kernel |
             +-------------+
               |    |    |
               |    |    |
             gmem gmem1 gmem2
               |    |    |
               A    B    C
```

The intention is to allow the accesses to the input and output matrices to use independent memory channels.

This can potentially increase memory-level parallelism compared with placing all accesses on a single interface.

---

# AXI-Lite Control Interface

Kernel arguments are exposed through an AXI-Lite control interface:

```cpp
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = in2 bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = M bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = P bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
```

This allows the host application to configure:

```text
Input A address
Input B address
Output C address
N
M
P
```

before starting the kernel.

---

# Local FPGA Buffers

The kernel creates local storage:

```cpp
uint512_dt v1_local[BUFFER_SIZE];
uint512_dt v2_local[BUFFER_SIZE];
uint512_dt result_local[BUFFER_SIZE];
```

These buffers are intended to reduce repeated accesses to global memory.

The conceptual data path is:

```text
Global Memory
     |
     v
+-------------+
| Local FPGA  |
| Buffer      |
+-------------+
     |
     v
Compute
     |
     v
+-------------+
| Output      |
| Buffer      |
+-------------+
     |
     v
Global Memory
```

Local FPGA storage can provide much lower access latency than repeatedly accessing external/global memory.

---

# HLS Dataflow

The design uses:

```cpp
#pragma HLS DATAFLOW
```

inside the row-processing loop.

Dataflow allows independent regions of the design to potentially execute concurrently when their dependencies permit it.

Conceptually:

```text
          +-------------+
          | Load A row  |
          +------+------+
                 |
                 v
          +-------------+
          | Load B data |
          +------+------+
                 |
                 v
          +-------------+
          | Computation |
          +------+------+
                 |
                 v
          +-------------+
          | Store C     |
          +-------------+
```

Rather than requiring the entire operation to behave as one sequential block, HLS can construct a streaming/pipelined architecture.

---

# Loop Pipelining

Memory loading operations use HLS pipelining:

```cpp
#pragma HLS pipeline
```

For example:

```cpp
for (int j = 0; j < M; j++) {
    #pragma HLS pipeline
    v1_local[j] = in1[i * N + j];
}
```

Pipelining allows iterations to overlap.

The goal is to reduce the initiation interval between successive iterations and keep the hardware datapath continuously utilized.

---

# Loop Unrolling

The dot-product computation contains:

```cpp
#pragma HLS UNROLL
```

on the innermost loop:

```cpp
for (int k = 0; k < M; k++) {
    #pragma HLS UNROLL

    ap_uint<32> tmp1 =
        v1_local[k].range(
            32 * (k + 1) - 1,
            k * 32
        );

    ap_uint<32> tmp2 =
        v2_local[k].range(
            32 * (k + 1) - 1,
            k * 32
        );

    tmpOut.range(
        32 * (k + 1) - 1,
        k * 32
    ) = tmp1 * tmp2;
}
```

Loop unrolling asks HLS to replicate the computation hardware rather than executing the loop iterations sequentially.

For:

```text
M = 64
```

a fully unrolled loop can potentially create up to:

```text
64 parallel multiplication operations
```

subject to resource availability and HLS scheduling.

This can significantly increase parallelism but also increases FPGA resource consumption.

---

# Packed Multiplication

Each 512-bit input contains sixteen 32-bit values.

The kernel extracts individual values:

```text
512-bit A word
+----+----+----+----+----+ ... +----+
| A0 | A1 | A2 | A3 | A4 | ... | A15|
+----+----+----+----+----+ ... +----+
 32b  32b  32b  32b  32b       32b
```

and similarly for B.

The computation then performs:

```text
A[k] × B[k]
```

for each element.

The resulting 32-bit products are packed back into:

```cpp
uint512_dt tmpOut;
```

using:

```cpp
tmpOut.range(...) = tmp1 * tmp2;
```

---

# Host Application

The host application is responsible for controlling the FPGA.

The program expects an `.xclbin` file:

```bash
./host <XCLBIN File>
```

For example:

```bash
./host matrix_mult.xclbin
```

---

# Host Data Generation

The host allocates four aligned buffers:

```cpp
std::vector<unsigned int, aligned_allocator<unsigned int>>
    source_in1(DATA_SIZE);

std::vector<unsigned int, aligned_allocator<unsigned int>>
    source_in2(DATA_SIZE);

std::vector<unsigned int, aligned_allocator<unsigned int>>
    source_hw_results(DATA_SIZE);

std::vector<unsigned int, aligned_allocator<unsigned int>>
    source_sw_results(DATA_SIZE);
```

The input values are initialized using:

```cpp
source_in1[i] = i;
source_in2[i] = i * i;
```

The CPU reference computation is then:

```cpp
source_sw_results[i] =
    source_in1[i] * source_in2[i];
```

This provides a software result against which the FPGA output can be checked.

---

# FPGA Programming

The host discovers available AMD/Xilinx accelerator devices using:

```cpp
auto devices = xcl::get_xil_devices();
```

The compiled FPGA binary is loaded using:

```cpp
auto fileBuf =
    xcl::read_binary_file(binaryFile);
```

and programmed onto the selected device.

The host reports:

```text
Trying to program device[...]
Device[...]: program successful!
```

when programming succeeds.

---

# OpenCL Buffers

The host creates three OpenCL buffers:

```cpp
buffer_in1
buffer_in2
buffer_output
```

using:

```cpp
CL_MEM_USE_HOST_PTR
```

This allows the OpenCL runtime to associate the device buffers with the aligned host-side memory.

---

# Data Transfer

Input data is transferred to the FPGA using:

```cpp
q.enqueueMigrateMemObjects(
    {buffer_in1, buffer_in2},
    0
);
```

The FPGA output is transferred back using:

```cpp
q.enqueueMigrateMemObjects(
    {buffer_output},
    CL_MIGRATE_MEM_OBJECT_HOST
);
```

The complete execution pipeline is therefore:

```text
             HOST CPU
                |
                |
         Input generation
                |
                v
        +---------------+
        | Host Buffers  |
        +-------+-------+
                |
          Host → FPGA
                |
                v
        +---------------+
        | Global Memory |
        |   gmem/gmem1 |
        +-------+-------+
                |
                v
        +---------------+
        | Local Buffers |
        +-------+-------+
                |
                v
        +---------------+
        | Parallel      |
        | Multipliers   |
        +-------+-------+
                |
                v
        +---------------+
        | Output Memory |
        |     gmem2     |
        +-------+-------+
                |
          FPGA → Host
                |
                v
        +---------------+
        | Result Check  |
        +---------------+
```

---

# Performance Measurement

The host uses `EventTimer` to measure different stages of the application.

Measured sections include:

```text
Allocate Memory in Host Memory
Fill the buffers
Software Mult run
OpenCL host code
Load Binary File to Alveo U200
Allocate Buffer in Global Memory
Set the Kernel Arguments
Copy input data to device global memory
Launch the Kernel
Copy Result from Device Global Memory to Host Local Memory
Compare the results
```

The timing report is printed using:

```cpp
et.print();
```

This makes it possible to separate:

* Host initialization time
* Software computation time
* FPGA programming time
* Buffer allocation time
* Host-to-device transfer time
* Kernel launch time
* Device-to-host transfer time
* Validation time

---

# CPU vs FPGA Validation

The host computes a software reference:

```cpp
for (int i = 0; i < DATA_SIZE; i++) {
    source_sw_results[i] =
        source_in1[i] * source_in2[i];
}
```

The FPGA output is then compared element-by-element:

```cpp
for (int i = 0; i < DATA_SIZE; i++) {
    if (source_hw_results[i] != source_sw_results[i]) {
        ...
    }
}
```

The program reports:

```text
TEST PASSED
```

when all values match.

Otherwise:

```text
TEST FAILED
```

is reported together with the first detected mismatch.

---

# Data Size

The host currently defines:

```cpp
#define DATA_SIZE 16384
```

This corresponds to:

```text
16,384 × 32-bit integers
```

or:

```text
65,536 bytes
```

of data per buffer.

The FPGA kernel, however, operates on 512-bit words, meaning one FPGA memory word contains:

```text
16 × 32-bit values
```

Therefore:

```text
16,384 / 16 = 1,024
```

512-bit words are required to represent 16,384 32-bit elements.

---

# FPGA Parallelism

The design combines several forms of parallelism.

## 1. Wide Memory Access

```text
512-bit memory word
        ↓
16 × 32-bit values
```

This increases the amount of data transferred per memory transaction.

---

## 2. Independent Memory Channels

```text
A → gmem
B → gmem1
C → gmem2
```

This separates the memory traffic associated with the two inputs and output.

---

## 3. Loop Pipelining

```cpp
#pragma HLS PIPELINE
```

allows consecutive loop iterations to overlap.

---

## 4. Loop Unrolling

```cpp
#pragma HLS UNROLL
```

replicates computation hardware to process multiple iterations concurrently.

---

## 5. Dataflow

```cpp
#pragma HLS DATAFLOW
```

allows independent regions of the kernel to overlap when possible.

---

# Optimization Strategy

The design can be viewed as a progression from a simple software implementation toward a highly parallel FPGA implementation:

```text
CPU implementation
       |
       v
Basic FPGA kernel
       |
       v
Local buffering
       |
       v
512-bit data width
       |
       v
Multiple AXI interfaces
       |
       v
Loop pipelining
       |
       v
Loop unrolling
       |
       v
Dataflow
       |
       v
Highly parallel FPGA architecture
```

Each optimization attempts to reduce one or more bottlenecks.

---

# Resource Trade-offs

FPGA optimization is not simply about maximizing parallelism.

Increasing the amount of parallel hardware can increase:

* LUT usage
* Flip-flop usage
* DSP usage
* BRAM usage
* Routing complexity
* Power consumption

For example, fully unrolling a loop with:

```text
M = 64
```

can potentially require many multiplication units.

Therefore, an important part of the project is finding the balance between:

```text
Performance
     ↕
Resource utilization
```

---

# Planned Experiments

The design can be evaluated under different configurations.

## Experiment 1 — Data Width

Compare:

```text
32-bit
128-bit
256-bit
512-bit
```

and measure:

* Memory throughput
* Kernel latency
* Resource utilization

---

## Experiment 2 — Loop Unrolling

Compare different unroll factors:

```text
No unrolling
×2
×4
×8
×16
×32
×64
```

and evaluate how performance scales.

---

## Experiment 3 — Pipeline Initiation Interval

Investigate:

```text
II = 1
II = 2
II = 4
...
```

and determine the effect on throughput.

---

## Experiment 4 — Memory Architecture

Compare:

```text
Single AXI interface
```

against:

```text
gmem
gmem1
gmem2
```

to investigate the impact of independent memory channels.

---

## Experiment 5 — Matrix Size

Evaluate different matrix dimensions:

```text
16 × 16
32 × 32
64 × 64
128 × 128
256 × 256
```

and analyze:

* Execution time
* Throughput
* FPGA resource usage
* Memory bandwidth requirements

---

# Metrics

Important metrics for the experiments include:

| Metric               | Description                          |
| -------------------- | ------------------------------------ |
| Kernel latency       | Time required for FPGA computation   |
| Total execution time | Complete host + FPGA execution time  |
| Host-to-device time  | Input transfer latency               |
| Device-to-host time  | Output transfer latency              |
| Throughput           | Amount of computation per second     |
| Clock frequency      | FPGA operating frequency             |
| Initiation Interval  | Distance between pipeline iterations |
| LUT utilization      | FPGA lookup-table usage              |
| FF utilization       | Flip-flop usage                      |
| DSP utilization      | Dedicated arithmetic resources       |
| BRAM utilization     | On-chip block RAM usage              |
| Memory bandwidth     | Data transferred per second          |

