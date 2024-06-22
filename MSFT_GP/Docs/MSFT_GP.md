# MSFT vs GP

## Goals
My current goal is to learn about stock price data and gaussian processes. Thus, I decided to try and fit Microsoft stock data using gaussian processes. At least, this was the original idea, as this won't work the way I thought it would (cf. [Analyses](#analyses)). I may decide to stick to the  <a href="/DataSets/Docs/Datasets.md">MSFT dataset</a> but switch to other methods. The goal is not to implement a competitive GP library. There are much better options out there (e.g. [2. in References](#references)).

## What are Gaussian Processes?
The textbook answer may be something like "A probability distribution over function spaces". In this repo I used it as a tool to fit training data and test the predictions against test data just using the assumed or estimated covariance structure of the data.

I won't go into any more detail as there is great source material available. Check out [1.](#references) and [3.](#references) in references.

## Implementation
The gp implementation [gaussian_process.py](../gaussian_process.py) is homebrew. It is mostly based on [1.](#references) (lectures 9 and 12 if I remember correctly). The rest is currently pretty much plot-tools and data-plumbing using pandas. 

## Usage
In this folder:
`python -m main [-h] [--mode MODE]`

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


