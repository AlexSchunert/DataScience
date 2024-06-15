### Initial Example
To generate the plot, run `python -m main --mode init_example`
<p align="center">
  <img src="resources/PriceHigh_19890601_19891231_HandTuned.png" alt="drawing" width="600"/>
  <br>
  <em>Figure 1: Initial example.</em>
</p>

* Highest microsoft stock price for June 1st and December 31st 1989
* Training data  <span style="color: green;">&#9733;</span> (80%), test data <span style="color: blue;">&#9679;</span> (20%)
* Green line indicates mean function $\mu$
* Standard deviation $\sigma_{\mu}$ indicated by red dashed lines and shaded grey area
* Radial basis function function with parameters (definition cf. [3.](#references))

<div style="margin-left: 30px;">

|Parameter                    |Value     |
|-----------------------------|----------|
|length-scale                 |10.0 days |
|output-scale                 |5.0 $     |
|$\sigma_{P}$: std of price   |0.01 $    |
</div>

It should be noted here that those parameters have been chosen to achieve a visually compelling result. If it's of any practical use remains to be seen.

 
