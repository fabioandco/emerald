---
title: Testing investment strategies accross markets? A good idea to increase robustness and protect from over-fitting
---

In the following lines I propose a robustenss test for systematic investment strategies that consists in testing an investment signal across different markets. 
Robustness tests are a key step of a healthy research process: they increase the burden of criteria that investments strategies have to pass before being deployed live and in this way they can reduce the likelihood of finding false investment strategies. 
In this post I will provide some evidence that testing an investment signal accross multiple markets makes sense also from a stastical standpoint.

**Intro** 

Researchers might have the tendency to test multiple hypothesis and only present the successful ones (this is known as selection bias). On the top of that, testing many hyptohesis under different settings can easilily lead to false discoveries (a strategy which appears successful in the backtest sample just because of noise in the data and not because of a true pattern that is exploited by the strategy).

A valid research process should therefore entail a batch of robustness tests once the researcher individuates a potential profitable investment strategy with strong economic underpinnings. Testing the investment signal across different markets consistues one of these robustess tests. In the next lines I show why this is a good idea from a statistical standpoint. In fact testing a signal on different markets reduces the probability of picking up false strategies (i.e. false positives).

