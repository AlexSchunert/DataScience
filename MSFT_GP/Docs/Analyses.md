### Initial Example
To generate the plot, run `python -m main --mode init_example`
<p align="center">
  <img src="resources/PriceHigh_19890601_19891231_HandTuned.png" alt="drawing" width="600"/>
  <br>
  <em>Figure 1: Initial example.</em>
</p>
Notes:

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

### One-day returns using adjusted closing price
Start with a plot of the timeseries of one-day returns of the adjusted closing prices
* Analyze return instead of price => timeseries is "closer" to stationarity => Use of standard signal processing methods possible
* One-day returns of adjusted closing price is used => Closing price is considered the most important stock price of the day

To generate the plot, run `python -m main --mode plot_return_ts` 

<p align="center">
  <img src="resources/OneDayReturns_ClosingAdj_TimeseriesFull.png" alt="drawing" width="600"/>
  <br>
  <em>Figure 2: One-day returns based on adjusted closing price.</em>
</p>
Notes:

* Unfortunately, the plot suggests very small to no temporal correlation of the returns. As far as my reading goes, this aligns with empirical findings from finance and is to be expected for a somewhat efficient market. 
* If there is no temporal correlation in the timeseries, using a gaussian process (or any other method) is pointless: 
  * The predictive covariance becomes diagonal with identical diagonal elements => representer weights are simply the scaled observations
  * Due to low correlation, the prediction does not even depend on the data and is simply given by the prior 
  * "Prediction" for training data yields a result between prior and original observation depending on the assumed noise level. 

Let's look at the data in more detail:

To generate the plot, run `python -m main --mode plot_return_full`

<p align="center">
  <img src="resources/OneDayReturns_ClosingAdj_AllPlotsFull.png" alt="drawing" width="600"/>
  <br>
  <em>Figure 3: Detailed look at one-day returns based on adjusted closing price.</em>
</p>

Notes:

* signal vs time is identical to Figure 1
* Autocorrelation shows the autocorrelation function of the signal using the function acovf from statsmodels.tsa.stattools to handle missing data (no trading at the weekend).
* PSD is the power spectral density estimated using a Lomb-Scargle periodogram taken from astropy.timeseries (again to handle data gaps). The cut-off frequency is set according to the median sampling of the data. 
* Histogram is simply the histogram of the returns
* We need to be a bit careful with the interpretation as acf- and psd- calculation require stationarity. 
  * While the mean of the data is most likely stationary, the variance appears to vary quite a bit. 
  * There seem to be clusters of higher and clusters of lower variance (=> volatility clustering?). 
  * We'll look at subsets of the data down the line to make sure nothing funny is going on.
* The result suggests that there are no temporal correlations in one-day returns:
  * Autocorrelation drops immediately to zero
  * PSD is flat(ish) => white noise

In order to make sure to not miss anything, it makes sense to look at some subsets of the data with constant variance. 

To generate the plot, run `python -m main --mode plot_return_full_subs`
<p align="center">
  <img src="resources/OneDayReturns_ClosingAdj_AllPlotsSubsLowVar.png" alt="drawing" width="400"/> &nbsp;&nbsp;&nbsp;  
    <img src="resources/OneDayReturns_ClosingAdj_AllPlotsSubsHighVar.png" alt="drawing" width="400"/>
  <br>
  <em>Figure 4: Detailed look at subsets of the data assumed stationary.</em>
</p>

Notes:
* Timeframe with low variance 1st January 1993 - 31st December 1995 (left)
* Timeframe with high variance 1st January 2000 - 31st December 2003 (right)
* Pretty much confirms results in Figure 3 => No autocorrelation in the data

**Bottom line:** Predicting one-day returns using GPs will not work due to the lack of temporal correlation. 

