## For data
import pandas as pd                     # `pip3 install pandas`
import numpy as np
## For plotting
import matplotlib.pyplot as plt         # `pip3 install matplotlib`
## For parametric fitting
from scipy import optimize              # `pip3 install scipy`

dtf = pd.read_csv("https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_canada/cases_timeseries_canada.csv", sep=",")
print(dtf.head())


# Columns:     date_report  cases  cumulative_cases
## convert date_report to datetime
dtf.date_report = pd.to_datetime(dtf.date_report, infer_datetime_format=True)


## Filter out the first spike because we're big lame smudgy boys
dtf_before = dtf[dtf['date_report'] <= pd.to_datetime('2020-03-08')]
dtf_after = dtf[dtf['date_report'] >= pd.to_datetime('2020-08-01')]

# dtf = pd.concat([dtf_before, dtf_after])
dtf = dtf_after

print(dtf.head())
print(dtf.tail())

## Change date_report to be the index column
dtf = dtf.reset_index().set_index('date_report')

## Drop Provice/Country and 'index' columns
dtf = dtf.drop(['province','index'], axis=1)


print(dtf.head())
print(dtf.tail())



'''
Logistic function: f(x) = capacity / (1 + e^-k*(x - midpoint) )
'''
def logistic_f(X, c, k, m):
    y = c / (1 + np.exp(-k*(X-m)))
    return y
## optimize from scipy
logistic_model, cov = optimize.curve_fit(logistic_f, 
                                xdata=np.arange(len(dtf["cumulative_cases"])), 
                                ydata=dtf["cumulative_cases"].values, 
                                maxfev=10000, 
                                p0=[np.max(dtf["cumulative_cases"])*1.5, 1, 1]
                                # p0 = None
                                )
## print the parameters
print(logistic_model)


'''
Gaussian function: f(x) = a * e^(-0.5 * ((x-μ)/σ)**2)
'''
def gaussian_f(X, a, b, c):
    y = a * np.exp(-0.5 * ((X-b)/c)**2)
    return y
## optimize from scipy
gaussian_model, cov = optimize.curve_fit(gaussian_f,
                                xdata=np.arange(len(dtf["cases"])), 
                                ydata=dtf["cases"].values, 
                                maxfev=10000,
                                # p0=[1, np.mean(dtf["cases"]), 1]
                                p0 = None
                                )
## print the parameters
print(gaussian_model)


'''
Plot parametric fitting.
'''
def utils_plot_parametric(dtf, zoom=30, figsize=(15,5)):
    ## interval
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    dtf["conf_int_low"] = dtf["forecast"] - 1.96*dtf["residuals"].std()
    dtf["conf_int_up"] = dtf["forecast"] + 1.96*dtf["residuals"].std()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    ## entire series
    dtf["ts"].plot(marker=".", linestyle='None', ax=ax[0], title="Parametric Fitting", color="black")
    dtf["model"].plot(ax=ax[0], color="green")
    dtf["forecast"].plot(ax=ax[0], grid=True, color="red")
    ax[0].fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)
   
    ## focus on last
    first_idx = dtf[pd.notnull(dtf["forecast"])].index[0]
    first_loc = dtf.index.tolist().index(first_idx)
    zoom_idx = dtf.index[first_loc-zoom]
    dtf.loc[zoom_idx:]["ts"].plot(marker=".", linestyle='None', ax=ax[1], color="black", 
                                  title="Zoom on the last "+str(zoom)+" observations")
    dtf.loc[zoom_idx:]["model"].plot(ax=ax[1], color="green")
    dtf.loc[zoom_idx:]["forecast"].plot(ax=ax[1], grid=True, color="red")
    ax[1].fill_between(x=dtf.loc[zoom_idx:].index, y1=dtf.loc[zoom_idx:]['conf_int_low'], 
                       y2=dtf.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)
    plt.show()
    return dtf[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]


'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param f: function
    :param model: list of optim params
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_curve(ts, f, model, pred_ahead=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    X = np.arange(len(ts))
    fitted = f(X, model[0], model[1], model[2])
    dtf = ts.to_frame(name="ts")
    dtf["model"] = fitted
    
    ## index
    start=ts.index[-1]
    index = pd.date_range(start=start,periods=pred_ahead,freq=freq)
    index = index[1:]
    ## forecast
    Xnew = np.arange(len(ts)+1, len(ts)+1+len(index))
    preds = f(Xnew, model[0], model[1], model[2])
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
    
    ## plot
    utils_plot_parametric(dtf, zoom=zoom)
    return dtf


preds_cumulative_cases = forecast_curve(dtf["cumulative_cases"], logistic_f, logistic_model, pred_ahead=2000, freq="D", zoom=7)


preds_new_cases = forecast_curve(dtf["cases"], gaussian_f, gaussian_model, pred_ahead=500, freq="D", zoom=7)

