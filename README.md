# Background-Subtraction-for-Moving-Object-detection
A high-performance Python implementation of the Embedded Background Subtraction (EBGS) algorithm for moving object detection, based on the 2021 paper.

# Complete Technical Analysis of EBGS: A Full-Process Optimization-Based Background Subtraction Framework

## Overview and Problem Statement

The **Embedded Background Subtraction (EBGS)** paper addresses a critical challenge in computer vision: developing an efficient background subtraction algorithm that can run in real-time on resource-constrained embedded devices while maintaining high detection accuracy. The authors propose a **full-process optimization** approach that simultaneously optimizes all three components of background subtraction: background modeling, post-processing, and model updating.

## Complete System Architecture and Flow

### Core Design Philosophy

EBGS follows a **streamlined design methodology** with three key optimization principles:

1. **Computational Efficiency**: Minimize processing overhead across all stages
2. **Memory Optimization**: Reduce memory footprint for embedded deployment  
3. **Accuracy Preservation**: Maintain detection quality despite optimizations

The complete system flow is illustrated in Figure 1 of the paper, showing the integration of all components.

## Detailed Technical Components

### 1. Background Modeling and Foreground Detection

#### 1.1 Contracted Codebook Model

**Core Concept**: Each pixel maintains its own **contracted codebook** $$ C = \{c_1, c_2, ..., c_L\} $$ where each codeword $$ c_k $$ contains:

- **Color channels**: $$ R, G, B $$ intensity values
- **Grayscale value**: $$ g $$  
- **Weight**: $$ w $$ (importance/frequency)
- **Last update time**: $$ \theta $$

**Initialization Process**:
```
For each pixel (x,y):
   Create codebook C with first frame values
   Initialize weight w = 1/S (S = training frames)
   Set timestamp θ = 1
```

**Weight Update Mechanism**:
$$
\omega_{k,t} = \omega_{k,t-1} + \frac{1}{S}
$$

**Parameter Update Formula**:
$$
\mu_{k,t+1} = (1-\lambda)\mu_{k,t} + \lambda X_t
$$

where $$ \lambda $$ is the learning rate and $$ X_t $$ represents pixel values at frame $$ t $$.

#### 1.2 Codeword Selection and Pruning

**Dynamic Codeword Management**:
$$
L = \arg\min_n \left[\sum_{i=1}^n \omega_i > 0.7\right]
$$

**Key Constraints**:
- Maximum 5 codewords per pixel (memory limitation)
- Stationary foreground absorption: Objects static for >20 frames become background
- Replacement strategy: Remove oldest codeword when limit exceeded

#### 1.3 Dual-Threshold Detection Strategy

**Hierarchical Detection Process**:

**Stage 1 - Grayscale Detection** (Lower threshold $$ T_L = 10 $$):
```
if |grayscale_current - grayscale_model| ≤ T_L:
    classify as background
else:
    proceed to Stage 2
```

**Stage 2 - Color Channel Verification** (Higher threshold $$ T_H = 17 $$):
$$
MAP = \begin{cases}
0, & \text{if } |X_i - \mu_i| \leq T_H \text{ for } i \in \{R,G,B\} \\
1, & \text{otherwise}
\end{cases}
$$

**Simplified Update Rule**:
$$
\mu_{k,t+1} = \begin{cases}
\mu_{k,t} + 1, & \text{if } T_L 25 FPS on 240×320 resolution SBC

### Challenging Scenario Performance

**Dynamic Backgrounds**: Snow, fountain sequences demonstrate superior noise handling through AIMD filtering robustness.

**Illumination Changes**: Fast codeword regeneration enables rapid adaptation to lighting variations.

**Ghost Object Handling**: 20-frame stationary threshold effectively absorbs static objects into background model.

**Camera Jitter**: Block-based update mechanism provides stability against minor camera movements.

## Key Innovations and Contributions

### 1. Full-Process Optimization
**Simultaneous optimization** of all three BGS components rather than isolated improvements.

### 2. AIMD Theoretical Framework  
**First rigorous mathematical analysis** of AIMD filtering using Markov chain theory with proven performance bounds.

### 3. Adaptive Random Update
**Block-based circular queue mechanism** providing computational efficiency with maintained randomness properties.

### 4. Embedded-Optimized Design
**Resource-aware architecture** specifically designed for general-purpose embedded devices with limited computational and memory resources.

## Implementation Details and Practical Considerations

### Hardware Requirements
- **Target Platform**: 900MHz quad-core ARM CPU, 1GB RAM
- **Memory Footprint**: Reduced through 5-codeword limit and circular queue reuse
- **Power Consumption**: Optimized for battery-powered outdoor deployment

### Parameter Sensitivity
- **Threshold Selection**: $$ T_L = 10, T_H = 17 $$ empirically determined
- **AIMD Parameter**: $$ \tau = 7 $$ balances accuracy and computational cost
- **Update Bounds**: $$ \alpha \in [0.05, 1.0] $$ prevents model stagnation

### Real-World Deployment
The paper demonstrates **practical applicability** through extensive testing on challenging scenarios including:
- Indoor/outdoor surveillance
- Variable lighting conditions  
- Dynamic background elements
- Multiple object scenarios
- Camera movement tolerance

This comprehensive framework represents a significant advancement in embedded computer vision by providing both theoretical rigor and practical efficiency for real-time background subtraction applications.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/48520623/5f9af7f0-9f54-4f02-aa3e-9efa9a7d6a0b/EBGS_Paper.pdf
