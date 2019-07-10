### Learning Loss Tensorflow
Simple Tensorflow implementation of "Learning Loss for Active Learning" (CVPR Oral 2019)

### Abstract
<p align="justify">
  The performance of deep neural networks improves with more annotated data. The problem is that the budget for annotation is limited. One solution to this is active learning, where a model asks human to annotate data that it perceived as uncertain. A variety of recent methods have been proposed to apply active learning to deep networks but most of them are either designed specific for their target tasks or computationally inefficient for large networks. In this paper, we propose a novel active learning method that is simple but task-agnostic, and works efficiently with the deep networks. We attach a small parametric module, named "loss prediction module," to a target network, and learn it to predict target losses of unlabeled inputs. Then, this module can suggest data that the target model is likely to produce a wrong prediction. This method is task-agnostic as networks are learned from a single loss regardless of target tasks. We rigorously validate our method through image classification, object detection, and human pose estimation, with the recent network architectures. The results demonstrate that our method consistently outperforms the previous methods over the tasks.
  
</p>

This repository provides a Tensorflow implementation of Learning Loss as described in the paper

> Learning Loss for Active Learning
> Donggeun Yoo, In So Kweon.
> CVPR Oral, 2019.
> [[Paper]](https://arxiv.org/abs/1905.03677)

### Requirements
The codebase is implemented in Python [TODO] package versions used for development.
'''

'''

### Options

### Model Options

### Examples


## Summary
### dataset
* [tiny_imagenet](https://tiny-imagenet.herokuapp.com/)
* cifar10, cifar100, mnist, fashion-mnist in `keras` (`pip install keras`)

### Train
* python main.py --phase train --dataset tiny --res_n 18 --lr 0.1

### Test
* python main.py --phase test --dataset tiny --res_n 18 --lr 0.1

