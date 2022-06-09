from scipy.stats import binned_statistic
import numpy as np


def sf2(t, y, yerr, bins):
    """Calculate structure function squared

    Short description goes here

    Parameters
    ----------
    t : `np.array` [`float`]
        Times at which the measurment was conducted
    y : `np.array` [`float`]
        Measurment values
    yerr : `np.array` [`float`]
        Measurment errors
    bins:  `np.array` [`float`]

    Returns
    ----------
    SF, bin_edge :

    Notes
    ----------

    TODO:
    ----------
    - allow user to not specify bins - automatically assume ``reasonable bins''
    - allow user to not specify times - assume equivdistant times
    - allow multiple inputs, with same <t> at once
    - ability to create SF2 from multiple lightcurves at once (ensamble)
    - allow for different definition of SF2

    """

    ## compute dt and dm for all gaps
    # dt
    dt_matrix = t.reshape((1, t.size)) - t.reshape((t.size, 1))
    dts = dt_matrix[dt_matrix > 0].flatten().astype(np.float16)

    # dm
    dm_matrix = y.reshape((1, y.size)) - y.reshape((y.size, 1))
    dms = dm_matrix[dt_matrix > 0].flatten().astype(np.float16)

    # err^2
    err2_matrix = yerr.reshape((1, yerr.size)) ** 2 + yerr.reshape((yerr.size, 1)) ** 2
    err2s = err2_matrix[dt_matrix > 0].flatten().astype(np.float16)

    ## SF for each pair of observations
    sfs = dms ** 2 - err2s

    # SF for at specific dt
    # the line below will throw error if the bins are not covering the whole range
    SFs, bin_edgs, _ = binned_statistic(dts, sfs, "mean", bins)

    return SFs, (bin_edgs[0:-1] + bin_edgs[1:]) / 2
