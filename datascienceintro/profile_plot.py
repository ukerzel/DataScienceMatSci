import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pplot(y_true, y_hat, ax, n_bins=20, yerr='var'):

    """ create a profile plot from the pair of predictions and true values.
        The profile plot is created by binning the predictions (these form the x-axis) and then 
        calculate the mean of the true values (these are on the y-axis). Ideally, the prediction should all
        be on the diagonal line, showing an unbiased estimator.

        To determine whether or not the mean true y values are compatible with the diagonal line, 
        we need a dispersion metric. There is no single choice and we can use:
        std: the stanard deviation of the distribution of y
        var: the variance
        RMS: the root-mean-square
        or others.

        Example usage:
        fig, ax = plt.subplots(1, 1)
        out = profile_plot.pplot(y_test, y_hat, n_bins=10, yerr='var', ax=ax)
        plt.show()

    Args:
        y_true: true values (label / target)
        y_hat: predictions
        n_bins: number of bins for the profile plot
        yerr: dispersion metric
        ax: axis of the matplotlib object on which the figure is drawn. This needs to be defined outside the funciton.

    Returns:
        matplotlib plot: Resulting profile plot as matplotlib object
    """

    # make a choice:
    # yerr: std. error on the mean
    # std: width of the distribution in y.
    # RMS: RMS of the distribution in y
    # depends on what is relevant.
    #

    x_min = np.min(y_hat) - (np.max(y_hat)-np.min(y_hat))/20.0 # remove a bit to be able to show the last bin
    x_max = np.max(y_hat) + (np.max(y_hat)-np.min(y_hat))/20.0 # add a bit to be able to show the last bin
    #n_bins = 20

    df = pd.DataFrame({'x': y_hat, 'y': y_true})

    # determine bins, +1 as we neeed the bin edge
    bins = np.linspace(x_min, x_max, n_bins+1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]


    df['bin'] = np.digitize(y_hat, bins=bins)
    binned = df.groupby('bin')

    def RMS(x):
        return np.sqrt(sum(x**2/len(x)))

  

    result = binned['y'].agg(['mean', 'std', 'var', RMS, 'count'])



    for i in range(1,n_bins+1):
        if i not in result.index:
            result.loc[i] = [np.nan,np.nan, np.nan, np.nan, np.nan]
        
    #make sure that we have each a value in each bin and no NaNs
    #result = result.fillna(0.0)

    result = result.sort_index()

    result['x'] = bin_centers
    result['x_err'] = bin_width / 2


    out = result.plot(
        x='x',
        y='mean',
        xerr='x_err',
        yerr=yerr,
        linestyle='none',
        capsize=0,
        color='black',
        legend=None,
        ax=ax
    )
    #draw diagonal line
    plt.plot([x_min, x_max], [x_min, x_max], 'k-', lw=2)
    plt.xlim([x_min, x_max])
    plt.ylim([x_min, x_max])
    plt.xlabel('prediction', fontsize = 20)
    plt.ylabel('true value', fontsize = 20)
    plt.title('Profile Plot')
    plt.tight_layout()

    return(out)