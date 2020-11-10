import numpy as np
from numba import njit

__all__ = ['weno4']
__version__='1.1.1'

def weno4(xs, xp, fp, left=None, right=None, extrapolate=False, assumeSorted=False, forceQuadratic=False):
    '''
    One-dimensional interpolation using the fourth-order Weighted Essentially
    Non-Oscillatory (WENO) scheme detailed in Janett et al (2019)
    (https://ui.adsabs.harvard.edu/abs/2019A%26A...624A.104J/abstract)


    Returns the WENO4 interpolant to the function given at discrete points
    (`xp`, `fp`), evaluated at each point in `x`. In the first and last
    intervals a one-sided quadratic interpolant is used, and so the
    non-oscillatory properties of WENO will not be present here. If this
    behaviour is required throughout the entirety of the input, then
    appropriate boundary conditions need to be determined and applied to data
    before interpolation. If the data is of length 3 then a single quadratic
    or linear interpolant is used (depending on the value of
    `forceQuadratic`), and a linear interpolant if the data is length 2.

    Parameters
    ----------
    xs : array_like
        The x-coordinates at which to compute the interpolant.

    xp : 1D array_like
        The x-coordinates at which the data is defined.

    fp : 1D array_like
        The values of the interpolated function at the data points `xp`.

    left : Optional[float]
        Value to use for `x < xp[0]`, by default `fp[0]`. Cannot be set with
        `extrapolate=True`.

    right : Optional[float]
        Value to use for `x > xp[-1]`, by default `fp[-1]`. Cannot be set
        with `extrapolate=True`

    extrapolate : bool
        Whether to extrapolate the function outside of `xp` instead of simply
        using the values of `left` and `right`. Extrapolation is performed by
        using the quadratic interpolating function computed for the first and
        last intervals. Default = False.

    assumeSorted : bool
        If True `xp` and `fp` are assumed to be provided in ordered such that
        `xp` is _strictly_ monotonically increasing. Otherwise these arrays
        are first sorted. Default = False.

    forceQuadratic : bool
        If True and `xp` and `fp` are of length 3 then a quadratic interpolant is used. Otherwise a linear interpolant is used. Default = False.

    Returns
    -------
    fs : ndarray
        The interpolated values, coresponding to each `x` in `xs`.

    Raises
    ------
    ValueError
        If `xp` and `fp` have different shapes
        If `xp` or `fp` are not one-dimensional
        If `extrapolate=True` and a value is set for `left` or `right`
        If `xp` and `fp` contain fewer than two elements
    '''
    xs = np.asarray(xs)
    xp = np.asarray(xp)
    fp = np.asarray(fp)

    if xp.shape != fp.shape:
        raise ValueError('xp and fp must have the same shape.')

    if xp.ndim != 1:
        raise ValueError('xp and fp must be one-dimensional.')

    if not assumeSorted:
        order = np.argsort(xp)
        xp = np.ascontiguousarray(xp[order])
        fp = np.ascontiguousarray(fp[order])

    Ngrid = xp.shape[0]
    if Ngrid < 2:
        raise ValueError('xp and fp are too short to interpolate.')
    if Ngrid == 2 or (Ngrid == 3 and not forceQuadratic):
        return np.interp(xs, xp, fp, left=left, right=right)

    return weno4_impl(xs, xp, fp, left=left, right=right, extrapolate=extrapolate)

@njit(cache=True)
def weno4_impl(xs, xp, fp, left=None, right=None, extrapolate=False):
    Ngrid = xp.shape[0]
    Eps = 1e-6
    fs = np.zeros_like(xs)
    xsFlat = xs.reshape(-1)
    fsFlat = fs.reshape(-1)

    if extrapolate and (left is not None or right is not None):
        raise ValueError('Cannot set both extrapolate and values for left and right.')

    if left is None:
        left = fp[0]
    if right is None:
        right = fp[-1]

    # NOTE(cmo): We store the previous index for which beta was computed to
    # avoid recalculation if possible. This can never be negative, so we
    # initialise it there to ensure it's computed the first time.
    prevBetaIdx = -1
    for idx, x in enumerate(xsFlat):
        # NOTE(cmo): Find i s.t. x \in [x_i, x_{i+1}).
        i = np.searchsorted(xp, x, side='right') - 1

        if x < xp[0]:
            if not extrapolate:
                fsFlat[idx] = left
                continue

            # NOTE(cmo): Put us at i == 0, extrapolating using the
            # quadratic defined over {x_0, x_1, x_2}.
            i = 0

        elif x > xp[-1]:
            if not extrapolate:
                fsFlat[idx] = right
                continue

            # NOTE(cmo): Put us at i == Ngrid - 1, extrapolating using the
            # quadratic defined over {x_{-3}, x_{-2}, x_{-1}}.
            i = Ngrid - 1

        # NOTE(cmo): For the last interval, it is valid to effectively use
        # inclusive on both ends. This also helps with extrapolation. We lose a
        # slight bit of efficiency for the case when x[i] == xp[-1], but this
        # is minor in real world problems.
        if i == Ngrid - 1:
            i -= 1

        if i == 0:
            xi = xp[i]
            xip = xp[i+1]
            xipp = xp[i+2]

            hi = xip - xi
            hip = xipp - xip

            yi = fp[i]
            yip = fp[i+1]
            yipp = fp[i+2]

            q3 = yi * ((x - xip) * (x - xipp)) / (hi * (hi + hip))
            q3 -= yip * ((x - xi) * (x - xipp)) / (hi * hip)
            q3 += yipp * ((x - xi) * (x - xip)) / ((hi + hip) * hip)

            fsFlat[idx] = q3
            continue
        elif i == Ngrid - 2:
            xim = xp[i-1]
            xi = xp[i]
            xip = xp[i+1]

            him = xi - xim
            hi = xip - xi

            yim = fp[i-1]
            yi = fp[i]
            yip = fp[i+1]

            q2 = yim * ((x - xi) * (x - xip)) / (him * (him + hi))
            q2 -= yi * ((x - xim) * (x - xip)) / (him * hi)
            q2 += yip * ((x - xim) * (x - xi)) / ((him + hi) * hi)

            fsFlat[idx] = q2
            continue

        xim = xp[i-1]
        xi = xp[i]
        xip = xp[i+1]
        xipp = xp[i+2]

        him = xi - xim
        hi = xip - xi
        hip = xipp - xip

        yim = fp[i-1]
        yi = fp[i]
        yip = fp[i+1]
        yipp = fp[i+2]

        # NOTE(cmo): Quadratics over substencils
        q2 = yim * ((x - xi) * (x - xip)) / (him * (him + hi))
        q2 -= yi * ((x - xim) * (x - xip)) / (him * hi)
        q2 += yip * ((x - xim) * (x - xi)) / ((him + hi) * hi)

        q3 = yi * ((x - xip) * (x - xipp)) / (hi * (hi + hip))
        q3 -= yip * ((x - xi) * (x - xipp)) / (hi * hip)
        q3 += yipp * ((x - xi) * (x - xip)) / ((hi + hip) * hip)

        # NOTE(cmo): Finite difference derivatives for smoothness indicators
        # If we are in the same [i, i+1) range as before, these can be reused.
        if i != prevBetaIdx:
            H = him + hi + hip
            yyim = - ((2*him + hi)*H + him*(him + hi)) / (him*(him + hi)*H) * yim
            yyim += ((him + hi)*H) / (him*hi*(hi + hip)) * yi
            yyim -= (him*H) / ((him + hi)*hi*hip) * yip
            yyim += (him*(him + hi)) / ((hi + hip)*hip*H) * yipp

            yyi = - (hi*(hi + hip)) / (him*(him + hi)*H) * yim
            yyi += (hi*(hi + hip) - him*(2*hi + hip)) / (him*hi*(hi + hip)) * yi
            yyi += (him*(hi + hip)) / ((him + hi)*hi*hip) * yip
            yyi -= (him*hi) / ((hi + hip)*hip*H) * yipp

            yyip = (hi*hip) / (him*(him + hi)*H) * yim
            yyip -= (hip*(him + hi)) / (him*hi*(hi + hip)) * yi
            yyip += ((him + 2*hi)*hip - (him + hi)*hi) / ((him + hi)*hi*hip) * yip
            yyip += ((him + hi)*hi) / ((hi + hip)*hip*H) * yipp

            yyipp = - ((hi + hip)*hip) / (him*(him + hi)*H) * yim
            yyipp += (hip*H) / (him*hi*(hi + hip)) * yi
            yyipp -= ((hi + hip) * H) / ((him + hi)*hi*hip) * yip
            yyipp += ((2*hip + hi)*H + hip*(hi + hip)) / ((hi + hip)*hip*H) * yipp

            # NOTE(cmo): Smoothness indicators
            beta2 = (hi + hip)**2 * (abs(yyip - yyi) / hi - abs(yyi - yyim) / him)**2
            beta3 = (him + hi)**2 * (abs(yyipp - yyip) / hip - abs(yyip - yyi) / hi)**2

            prevBetaIdx = i

        # NOTE(cmo): Linear weights
        gamma2 = - (x - xipp) / (xipp - xim)
        gamma3 = (x - xim) / (xipp - xim)

        # NOTE(cmo): Non-linear weights
        alpha2 = gamma2 / (Eps + beta2)
        alpha3 = gamma3 / (Eps + beta3)

        omega2 = alpha2 / (alpha2 + alpha3)
        omega3 = alpha3 / (alpha2 + alpha3)

        # NOTE(cmo): Interpolated value
        fsFlat[idx] = omega2 * q2 + omega3 * q3

    return fs

def test_weno4():
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d, PchipInterpolator
    plt.ion()

    Npoints = 20
    Seed = 17
    xp = np.sort(np.random.RandomState(seed=Seed).uniform(low=-1, high=1, size=Npoints))

    class Method:
        def __init__(self, interpolator, descriptor):
            self.interpolator = interpolator
            self.descriptor = descriptor

    def plot_function_test(ax, fn, xp, methods):
        fp = fn(xp)

        x = np.linspace(xp.min(), xp.max(), 10001)
        fRef = fn(x)
        ax.set_title(fn.__name__)
        ax.plot(xp, fp, 'o', label='Data')
        for method in methods:
            fMethod = method.interpolator(x, xp, fp)
            ax.plot(x, fMethod, label=method.descriptor)
        ax.plot(x, fRef, '--', label='True')

    def modified_heaviside(x):
        return np.where(x < 0, 0, 4.0)

    def exponential(x):
        return np.exp(1.5 * x)

    def gaussian(x):
        return 5 * (1 - np.exp(-4* x**2))

    def discontinuous_sine(x):
        return np.where(x < 0, 2 * np.sin(3*x) + 4, 2 * np.sin(3*x))

    methods = [
        Method(lambda x, xp, fp: interp1d(xp, fp, kind=3)(x), 'Cubic Spline'),
        Method(lambda x, xp, fp: PchipInterpolator(xp, fp)(x), 'PCHIP'),
        Method(lambda x, xp, fp: weno4(x, xp, fp), 'Weno4'),
    ]

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    plot_function_test(ax[0, 0], modified_heaviside, xp, methods)
    plot_function_test(ax[0, 1], exponential, xp, methods)
    plot_function_test(ax[1, 0], gaussian, xp, methods)
    plot_function_test(ax[1, 1], discontinuous_sine, xp, methods)
    ax[0, 0].legend()

if __name__ == '__main__':
    test_weno4()
