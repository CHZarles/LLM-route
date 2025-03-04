# 推理优化学习资料总结

## 【夯实基础】C++和并行编程

### 1. 基础书籍

- **现代 C++ 教程：高速上手 C++ 11/14/17/20**

  - 链接: [现代 C++ 教程](https://changkun.de/modern-cpp/)

  - 简介: 快速复习C++ 11/14/17/20的教程，适合有一定基础的同学。

### 2. 入门导论

- **并行计算概念**

  - 链接: [并行计算导论](https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial)
  - 简介: 介绍并行计算的基础概念，帮助快速了解整体概念。

### 3. 视频教程

- **C++中的高性能并行编程与优化**
  - 链接: [B站视频教程](https://www.bilibili.com/video/BV1fa411r7zp)
  - 简介: 重点学习的教程，涵盖Cmake, OpenMP, CUDA编程，访存优化等高阶知识。

### 4. 配套代码

- **代码和作业**
  - 链接: [课程代码和作业](https://github.com/parallel101/course)
  - 简介: 视频教程的配套代码和作业，建议跟着视频自己动手写一遍并完成作业。

### 5. 视频教程2（可选）

- **并行计算及应用**
  - 链接: [伯克利并行计算课程](https://sites.google.com/lbl.gov/cs267-spr2022)
  - 简介: 伯克利并行计算巨佬James Demmel和Kathy Yelick的课程，适合有余力的同学深入学习。

### 6. GPU算子开发

- **9个tutorials**
  - 链接: [Triton教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
  - 简介: 学习Triton框架的官方文档，完成9个tutorials的常见算子开发入门。

### 7. 优秀资源

- **高性能GPU矩阵乘法**
  - 链接: [高性能GPU矩阵乘法](https://zhuanlan.zhihu.com/p/531498210)
  - 简介: 文章详细讲解了矩阵乘法的优化过程，读完肯定受益匪浅。

### 8. 学习的重点

- C/C++
- 计算机架构
- 并行计算原理
- CUDA编程和优化
- GPU算子开发

## 【技术进阶】深度学习系统和AI编译器

### 1. 视频课程

- **陈天奇:MLC-机器学习编译**
  - 链接: [B站视频课程](https://space.bilibili.com/1663273796/channel/collectiondetail?sid=499979)
  - 简介: 学习TVM+AI模型编译技术，从导入Pytorch/TF模型到自动模型调优autoTVM。

### 2. 优秀资源

- **TVM和MLIR资料**
  - 链接: [TVM和MLIR资料](https://github.com/BBuf/tvm_mlir_learn)
  - 简介: 包含很多好的资源，了解LLVM和MLIR的基本原理，适合面试部署推理的工程师。

### 3. 学习的重点

- 机器学习工作流
- AI编译器前端和后端技术
- 模型转换，优化和压缩
- LLVM和MLIR原理
- TVM框架及全流程

## 本阶段作业和考试

1. 完成“C++中的高性能并行编程与优化”的章节作业，上传到学习空间。
2. 完成TVM对BERT模型的转换、编译以及优化，输出优化前后的性能比较。
