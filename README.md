# Incremental NDT-OMP algorithm (iNDT)

*Note: This README file has both Chinese version and English version, for English version please scroll down.*
<!-- I have offered  -->

## 一、简介

一种快速且优雅的 **增量式NDT算法** 开源实现 —— A Fast and Elegant Implementation of **Incremental NDT algorithm (iNDT)** based on ndt_omp。

众所周知，NDT（Normal Distribution Transform）算法是一种极为基础的点云配准算法，它将一帧点云（source）对齐到另一帧点云（target）上。
通常，NDT 算法需要先将 target 点云投影到排列整齐的体素（voxel）中，并对每一个体素内的点云拟合高斯分布；
而后，NDT对 source 点云中的每一个点寻找若干个近邻voxel，并用这些近邻voxel中的高斯分布来构建残差和雅可比，借助L-M优化算法实现梯度下降，获得最终的配准位姿。

传统NDT算法有两个耗时较高的环节可以被优化：
1. 每次NDT配准时都需要重新对target点云拟合高斯分布，即使当前target和上一次的target相差不大；
2. 每次重置target点云后，都需要重新生成kdtree，以满足为source点云中的点查找近邻voxel的需求。

针对上述问题，本仓库实现了增量式的NDT算法，`我们使用哈系表结构持有和维护所有的voxel，并在单个voxel内实现了高斯分布的增量式更新算法`，从而避免了重复性地重构体素地图，且无需构建kdtree即可实现近邻voxel查询，大大提升了效率。根据实验对比，在以NDT为基础的LIO应用中，iNDT(with omp) 的计算速度是传统 NDT(with omp) 的xx倍。


**未完待续 ...**

## 二、实验对比



## 三、编译与运行



## 四、应用示例（LIO & Localization）


<img src="data/screenshot.png" height="400pix" /><br>
Red: target, Green: source, Blue: aligned

## 五、相关仓库

- [ndt_omp](https://github.com/koide3/ndt_omp)
- [fast_gicp](https://github.com/SMRT-AIST/fast_gicp)


--- 
<!-- separating line  -->

Coming soon ...

## 1. Introduction

## 2. Experimental Results

## 3. Compile and Run

## 4. Application Examples (LIO & Localization)

## 5. Related packages
- [ndt_omp](https://github.com/koide3/ndt_omp)
- [fast_gicp](https://github.com/SMRT-AIST/fast_gicp)
