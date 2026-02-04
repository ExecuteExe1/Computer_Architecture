# Hardware/Software System Design – Lab 1: Vitis HLS Introduction
  
  ## Project Overview
   -This project involves designing a matrix multiplication accelerator (MATRIX_MUL) using Vitis HLS. The accelerator computes the product of two matrices A (n × m) and B (m × p), where each element is an 8-bit unsigned integer, and the result is stored as a 32-bit unsigned integer.
    The dimensions m, n, and p are powers of two, defined by parameters lm, ln, and lp respectively.

 ## 1. Implementation & Testbench
   -Write the C code for MATRIX_MUL in Vitis HLS according to the specifications, and create a corresponding testbench.
      
  ### Requirements
   - Use #define to set constants lm, ln, lp.
   - Compute m, n, p using bit-shift operations (1 << lm, etc.).
   - Do not use scanf() for initialization.
     
   - The testbench must:
    - Initialize matrices A and B with pseudo-random values in the range 0–255.
    - Compute the product using both:
    - A software reference implementation.
    - The hardware accelerator.
    - Compare results and print "Test Passed" if they match  
   
 ### Files
  - matrix_mul.cpp – Main accelerator function.
  - matrix_mul_tb.cpp – Testbench for validation.

## 2. Initial Synthesis
  -Run C Synthesis with default settings (no directives) and report the following for lm = ln = lp = 6

  ### Metric	Value
   - Estimated Clock Period	
   - Worst-Case Latency	
   - DSP48E Used	
   - BRAMs Used	
   - FFs Used	
   - LUTs Used
## 3. C/RTL Cosimulation
  -Run C/RTL Cosimulation to verify functional correctness and extract timing metrics.
     
 ### Metric	Value
  - Total Execution Time	
  - Min Latency	
  - Avg. Latency	
  - Max Latency

Results for lm = ln = lp = 6

## 4.Optimization with Directives
 -Apply Vitis HLS directives (e.g., ARRAY_PARTITION, PIPELINE, UNROLL) to minimize execution time.

### Tasks
  1.Experiment with different matrix dimensions and observe the effect when lm is fixed and ln/lp are varied. Explain why certain trends occur.
  2.Find the optimal configuration for lm = ln = lp = 6 and report:
   - Used directives and reasoning.
   - Optimized synthesis results (clock period, resource usage, timing).
  3.Calculate the speed-up of the optimized hardware design compared to the initial (unoptimized) version.


### Metric	Value
  - Estimated Clock Period	
  - DSP48E Used	
  - BRAMs Used	
  - FFs Used	
  - LUTs Used	
  - Total Execution Time	
  - Min Latency	
  - Avg. Latency	
  - Max Latency
  
  Optimized Results for lm = ln = lp = 6
