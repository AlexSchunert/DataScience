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

**Bottom line:** The parameters have been chosen to achieve a visually compelling result. If it's of any practical use remains to be seen. Implementation seems to work as result look plausible. 

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
  * The predictive covariance becomes diagonal => Representer-weights are simply the scaled observations. In case of existing temporal correlation, each representer-weight is a linear combination of observations with weights depending on the kernel function and the noise level. 
  * If there is no temporal correlation, the prediction does not even depend on the data and is simply given by the prior 
  * "Prediction" for training data yields a result between prior and original observation depending on the assumed noise level. 

Let's look at the data in more detail:

To generate the plot, run `python -m main --mode plot_return_full`

<p align="center">
  <img src="resources/OneDayReturns_ClosingAdj_AllPlotsFull.png" alt="drawing" width="600"/>
  <br>
  <em>Figure 3: Detailed look at one-day returns based on adjusted closing price.</em>
</p>

Notes:

* signal vs time is identical to Figure 2
* Autocorrelation shows the autocorrelation function of the signal using the function acovf from statsmodels.tsa.stattools to handle missing data (no trading at the weekend).
* PSD is the power spectral density estimated using a Lomb-Scargle periodogram taken from astropy.timeseries (again to handle data gaps). The cut-off frequency is set according to the median sampling of the data. 
* Histogram is simply the histogram of the returns.
* We need to be a bit careful with the interpretation as acf- and psd- calculation require stationarity. 
  * While the mean of the data is most likely stationary, the variance appears to vary quite a bit. 
  * There seem to be clusters of higher and clusters of lower variance (=> volatility clustering?). 
  * We'll look at subsets of the data down the line to make sure nothing funny is going on.
* The result suggests that there are no temporal correlations in one-day returns:
  * Autocorrelation drops immediately to zero
  * PSD is flat(ish) => white noise

Just to make sure, a detailed looks at two subsets of the data with constant variance. 

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

### Absolute values of one-day returns
Analyze absolute values of one-day returns:
* Volatility clustering: Large changes often follow large changes and small changes often follow small changes. => temporal correlation in the data
* Is it possible to predict absolute values of returns => proxy for volatility
* Why not squared returns?
  * Data would follow a $\chi^2$-distribution. 
  * For modelling with a GP later down the line, I rather have non-negative gaussian distributed data and deal with or accept the issues from having a non-negativity constraint, compared to $\chi^2$-distributed data.
  *  Not sure if this is the best course of action. According to ChatGPT using non-negative gaussian data leads to less issues in GPs provided that non-negativity is accounted for. $\chi^2$-distributed data violates the assumption inherent to GPs that likelihood function and prior probability density function (pdf) are gaussian. 

To generate the plot, run `python -m main --mode plot_return_ts --return_mode abs` 
<p align="center">
  <img src="resources/OneDayAbsReturns_ClosingAdj_TimeseriesFull.png" alt="drawing" width="600"/>
  <br>
  <em>Figure 5: Absolute values of one-day returns based on adjusted closing price.</em>
</p>

To generate the plot, run `python -m main --mode plot_return_full --return_mode abs` 

<p align="center">
  <img src="resources/OneDayAbsReturns_ClosingAdj_AllPlotsFull.png" alt="drawing" width="600"/>
  <br>
  <em>Figure 6: Detailed look at absolute values of one-day returns based on adjusted closing price.</em>
</p>

Notes:
* Like Figures 2 and 3 for the absolute values of the returns (adj. closing price)
* Autocorrelation and PSD suggest that there is some slight temporal correlation 
* Same as for Figures 2 and 3 => Variance over complete timeframe not constant => Look at subsets

To generate the plot, run `python -m main --mode plot_return_full_subs --return_mode abs`
<p align="center">
  <img src="resources/OneDayAbsReturns_ClosingAdj_AllPlotsSubsLowVar.png" alt="drawing" width="400"/> &nbsp;&nbsp;&nbsp;  
    <img src="resources/OneDayAbsReturns_ClosingAdj_AllPlotsSubsHighVar.png" alt="drawing" width="400"/>
  <br>
  <em>Figure 7: Detailed look at subsets of the data assumed stationary.</em>
</p>

Notes:
* Timeframes identical to Figure 4
* Timeframe with low variance 1st January 1993 - 31st December 1995 (left) => TFL
* Timeframe with high variance 1st January 2000 - 31st December 2003 (right) => TFH
* For TFH the temporal correlation is at least as "strong" as for complete timeseries (Figure 6)
* TFL shows hardly any temporal correlation

Look at two more subsets
To generate the plot, run `python -m main --mode plot_return_full_subs --return_mode abs`
<p align="center">
  <img src="resources/OneDayAbsReturns_ClosingAdj_AllPlotsSubsLowVar2.png" alt="drawing" width="400"/> &nbsp;&nbsp;&nbsp;  
    <img src="resources/OneDayAbsReturns_ClosingAdj_AllPlotsSubsHighVar2.png" alt="drawing" width="400"/>
  <br>
  <em>Figure 7: Detailed look at subsets of the data assumed stationary.</em>
</p>

Notes:
* Timeframe with low variance 1st January 2011 - 31st December 2012 (left) => TFL2
* Timeframe with high variance 1st January 2008 - 31st December 2009 (right) => TFH2
* TFL2 similar to TFL => Hardly any temporal correlation
* TFH2 shows quite significant temporal correlation compared to all other results

**Bottom line:**
* There seems to some temporal correlation in absolute values of returns
* Degree of temporal correlation varies within the timeseries

**Next:** 
* Use GP to predict absolute returns
* Estimate autocorrelation and use it to tune kernel hyperparameters
* Kernel function?

