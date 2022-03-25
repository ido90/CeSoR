# Efficient Risk-Averse Reinforcement Learning

This repo by Ido Greenberg implements the *Cross-entropy Soft-Risk* optimization algorithm (***CeSoR***) from the paper [Efficient Risk-Averse Reinforcement Learning]() by Greenberg and Mannor.

| <img src="https://github.com/ido90/CrossEntropySampler/blob/main/Images/CeSoR_results_summary.png" width="720"> |
| :--: |
| Summary of the results of 3 agents (risk-neutral PG, standard risk-averse [GCVaR](https://arxiv.org/abs/1404.3862), and our CeSoR) over 3 benchmarks. Top: the lower quantiles of the agent scores. Bottom: sample episodes. |

| <img src="https://github.com/ido90/CrossEntropySampler/blob/main/Images/CeSoR_driving_sample.gif" width="280"> |
| :--: |
| A sample episode of CeSoR in the Driving Game. The goal is to follow the leader as closely as possible without colliding. |

## Installation
`pip install -e .`

## Quick start - examples
* [`CEM_Example.ipynb`](https://github.com/ido90/CrossEntropySampler/blob/main/Examples/CEM_Example.ipynb): Minimal and explained example for interacting with the Cross Entropy module directly: implementation for a new family of distributions, running of a sampling process and analysis of the results.
* [`GuardedMazeExample.ipynb`](https://github.com/ido90/CrossEntropySampler/blob/main/Examples/GuardedMaze/GuardedMazeExample.ipynb), [`DrivingExample.ipynb`](https://github.com/ido90/CrossEntropySampler/blob/main/Examples/DrivingGame/DrivingExample.ipynb), [`ServersExample.ipynb`](https://github.com/ido90/CrossEntropySampler/blob/main/Examples/ServersAllocation/ServersExample.ipynb): End-to-end examples for CeSoR in 3 benchmarks: presentation of the benchmark, training and testing (or alternatively loading existing results), analysis and visual demonstrations.

## Background
In risk-averse Reinforcement Learning (RL), the goal is to optimize some risk-measure of the returns, which inherently focuses on the lower quantiles of the returns distribution.
This poses two difficulties: first, by focusing on certain quantiles we ignore some of the agent experience and thus reduce the sample efficiency. Second, ignoring the higher quantiles specifically leads to *blindness to success*: the optimizer is not exposed at all to beneficial behaviors of the agent.
To overcome these challenges, we present *CeSoR* - Cross-entropy Soft-Risk optimization algorithm. CeSoR leverages the Cross Entropy method to sample the lower quantiles over the environment conditions (minimizing over *epistemic* uncertainty); while using soft risk-level scheduling to expose the optimizer to the higher quantiles of the agent performance (miximizing over *aleatoric* uncertainty).
CeSoR works on top of the standard policy gradient algorithm and can be applied to various models including neural networks.
On benchmarks of maze navigation, autonomous driving and computational resources allocation, we show that CeSoR achieves better risk-measures than standard methods of both risk-neutral and risk-averse policy gradient, and sometimes works even when the standard risk-averse policy gradient completely fails.

## Algorithm
<img src="https://github.com/ido90/CrossEntropySampler/blob/main/Images/CeSoR_algorithm.png" width="400">
