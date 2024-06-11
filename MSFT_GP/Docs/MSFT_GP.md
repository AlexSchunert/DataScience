# MSFT vs GP

## Introduction
**Add motivation here**

<img src="resources/PriceHigh_19890601_19891231_HandTuned.png" alt="drawing" width="600"/>

The above example shows the highest microsoft stock price (cf. <a href="/DataSets/Docs/Datasets.md">Datasets</a>) for each day between June 1st and December 31st 1989 (<span style="color: blue;">&#9679;</span> and <span style="color: green;">&#9733;</span>). Using 80% of these data a Gaussian Process (GP) has been trained whose mean function $\mu$ is indicated by the green line.  Those data used for training (GP has been conditioned on these data) is referred to as training data and is shown as <span style="color: green;">&#9733;</span>. The complementary dataset, for which only predictions are made, is referred to as test data and indicated by <span style="color: blue;">&#9679;</span>-symbols. 
The fit between model and data looks quite promising considering the fact that the formulation of the model is quite simple. Apart from the choice of the kernel function and its parameters, only assumptions about the standard deviation of the data is necessary to achieve the fit

**Fit data and leave large gap in between with parameters==plot**
|Parameter                    |Value     |
|-----------------------------|----------|
|l: length-scale              |10.0 days |
|$\sigma_{rbf}$: output-scale |5.0  days |
|$\sigma_{P}$: std of price   |0.01$     |

$K(x,x')=exp\left(-\frac{\|x-x'\|_2}{2\sigma_{rbf}^2}\right)$


*The green line indicates the mean function $\mu$ of a Gaussian Process (with radial basis function kernel) conditioned on 80% of these data. The standard deviation $\sigma$ of the mean $\mu$ is displayed by the dashed lines.*


## References
1. [Probabilistic_ML](https://github.com/philipphennig/Probabilistic_ML)*: "Probabilistic Machine Learning" Course at the University of Tübingen, Summer Term 2023, Philipp Hennig. Under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.](https://creativecommons.org/licenses/by-nc-sa/4.0/)* Lecture provides a very deep introduction into Gaussian Processes and is complemented by youtube-videos.   

## Possible improvements
* Estimation of data standard deviation from data
* Windowed Fourier-Transform on return-data (stationary for certain periods?) => Estimation of covariance function
* Different kernel-functions 



