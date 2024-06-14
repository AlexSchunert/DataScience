# MSFT vs GP
## Table of Contents

1. [Introduction](#introduction)
2. [References](#references)


## Goals
My current goal is to learn about stock price data and gaussian processes. Thus, I decided to try and fit microsoft stock data using gaussian processes. At least, this was the original idea, as this won't work the way I thought it would (cf. [Analyses](#analyses)). I may decide to stick to the  <a href="/DataSets/Docs/Datasets.md">MSFT dataset</a> but switch to other methods. The goal is not to implement a competetice GP library. I am aware that there are existing implementations that are much better than my take (e.g. 2. in [References](#references)).

## What are Gaussian Processes?

The theroretical answer would be: A probability distribution over function spaces. My anwer would be: A tool to fit training data and predict for test data just using the assumed or estimated convariance structure of the data.

I won't go into any more detail as there is great source material available. Check out [1.](#references) and [3.](#references) in references.

## Implementation
The gp implementation [gaussian_process.py](../gaussian_process.py) is homebrew. The implementation is mostly based on [1.](#references) (lectures 9 and 12 if aI remember correctly). The rest is currently pretty much plot-tools and data-plumbing using pandas. 

## Usage
I haven't managed to write a command line parser yet. There is an 

## Analyses
A log of the [Analyses](Analyses.md) done so far.


## References
1. [Probabilistic_ML](https://github.com/philipphennig/Probabilistic_ML)*: "Probabilistic Machine Learning" Course at the University of Tübingen, Summer Term 2023, Philipp Hennig. Under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.](https://creativecommons.org/licenses/by-nc-sa/4.0/)* 
**Description**: Lecture provides a very deep introduction into Gaussian Processes and is complemented by youtube-videos. Highly recommend that for a deep dive.
2. [GP in scikit-learn](https://scikit-learn.org/stable/modules/gaussian_process.html)*: "Documentation of Gaussian processes in scikit-learn", v1.5.0, [scikit-learn core team](https://scikit-learn.org/dev/about.html#authors), Under [BSD 3-Clause License](https://opensource.org/license/bsd-3-clause)* 
**Description**: Implementation of Gaussian Processes in scikit-learn. I might use that in the future.
3. [GP by Mutual Information](https://www.youtube.com/watch?v=UBDgSHPxVME&t=432s)*: "Gaussian Processes" youtube-video, 23.08.2021, DJ Rich, Under Standard YouTube License.* Video provides a quick and easily accessible introduction into GPs. 
**Description**: A brief but through introduction. I'd recommend to start here. I used the definition of the radial basis function given there.    

<!--
## Possible improvements
* Estimation of data standard deviation from data
* Different kernel-functions 
* Estimation of covariance function
-->


