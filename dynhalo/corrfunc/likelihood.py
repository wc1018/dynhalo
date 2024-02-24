from typing import List, Tuple, Union

import numpy as np

from dynhalo.corrfunc.model import (error_func_pos_decr, power_law,
                                    rho_orb_model, xi_inf_model, xihm_model)


def orb_lnlike(
    pars: Union[List[float], Tuple[float, ...]],
    data: Tuple[np.ndarray]
) -> float:
    # Check parameter priors
    r_h, alpha, a, logd = pars
    out_of_prior_neg = any([p < 0 for p in (r_h, alpha, a)])
    out_of_prior_logd = (-4 > logd) & (logd > 0)
    if out_of_prior_neg or out_of_prior_logd:
        return -np.inf

    # Unpack data
    x, y, covy, morb = data
    delta = np.power(10., logd)

    # Reduce dynamical range of y and covy by multiplying by x^2
    x2 = x**2

    # Compute model deviation from data.
    d = x2 * (y - rho_orb_model(r=x, morb=morb, r_h=r_h, alpha=alpha, a=a))

    # Add regularization as a percent error to the covariance in quadrature.
    cov = np.outer(x2, x2) * (covy + np.diag((delta*y)**2))

    # Compute chi2
    chi2 = np.dot(d, np.linalg.solve(cov, d))

    # Compute the log of the determinant of the covariance. It is highly likely
    # to fall below the numerical precision of `float` since the covariances
    # tend to be very small. To avoid this we scale the covariance by the
    # maximum value squared of the y's. The scaling value is not relevant.
    # Return the negative of the determinant.
    scale = np.max(y)
    log_det_cov = len(d) * np.log(scale) + np.log(np.linalg.det(cov / scale))

    return -chi2 - log_det_cov


def orb_smooth_lnlike(
    pars: Union[List[float], Tuple[float, ...]],
    data: Tuple[np.ndarray]
) -> float:
    # Check parameter priors
    rh_p, rh_s, alpha_p, alpha_s, a, logd = pars
    out_of_prior_neg = any([p < 0 for p in (rh_p, rh_s, alpha_p, a)])
    out_of_prior_pos = any([p > 0 for p in (alpha_s, )])
    out_of_prior_logd = (-4 > logd) & (logd > 0)
    if out_of_prior_neg or out_of_prior_pos or out_of_prior_logd:
        return -np.inf

    # Unpack data and evaluate parameters
    x, y, covy, mask, morb, m_pivot = data
    delta = np.power(10., logd)
    r_h = power_law(morb / m_pivot, p=rh_p, s=rh_s)
    alpha = power_law(morb / m_pivot, p=alpha_p, s=alpha_s)

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for n_bin in range(y.shape[0]):
        # Reduce dynamical range of y and covy by multiplying by x^2
        x2 = x[mask[n_bin]]**2
        # Compute model deviation from data.
        d = x2 * (y[n_bin, mask[n_bin]] - rho_orb_model(r=x[mask[n_bin]],
                  morb=morb[n_bin], r_h=r_h[n_bin], alpha=alpha[n_bin], a=a))
        # Add regularization as a percent error to the covariance in quadrature.
        cov = np.outer(x2, x2) * (covy[n_bin, mask[n_bin], :][:, mask[n_bin]]
                                  + np.diag((delta * y[n_bin, mask[n_bin]])**2))

        # Compute chi2
        chi2 = np.dot(d, np.linalg.solve(cov, d))

        # Compute the log of the determinant of the covariance.
        scale = np.max(y[n_bin])
        log_det_cov = len(d) * np.log(scale) + \
            np.log(np.linalg.det(cov / scale))

        lnlike -= chi2 + log_det_cov
    return lnlike


def inf_lnlike(
    pars: Union[List[float], Tuple[float, ...]],
    data: Tuple[np.ndarray]
) -> float:
    # Check parameter priors
    bias, eta, gamma, r_inf, mu, logd = pars
    out_of_prior_neg = any([p < 0 for p in (eta, gamma, r_inf, mu)])
    out_of_prior_lrg = any([p > 10 for p in (bias, eta, gamma, r_inf, mu)])
    out_of_prior_bias = bias < 1
    out_of_prior_logd = (-4 > logd) & (logd > 0)
    if out_of_prior_neg or out_of_prior_lrg or out_of_prior_bias or out_of_prior_logd:
        return -np.inf

    # Unpack data
    x, y, covy, r_h, xi_large_call = data
    delta = np.power(10., logd)

    # Reduce dynamical range of y and covy by multiplying by x^2
    x2 = x**2

    # Compute model deviation from data.
    d = x2 * (y - xi_inf_model(r=x, r_h=r_h, bias=bias, xi_large=xi_large_call,
              eta=eta, gamma=gamma, r_inf=r_inf, mu=mu))

    # Add regularization as a percent error to the covariance in quadrature.
    cov = np.outer(x2, x2) * (covy + np.diag((delta*y)**2))

    # Compute the chi2
    chi2 = np.dot(d, np.linalg.solve(cov, d))

    # Compute the log of the determinant of the covariance.
    scale = np.max(y)
    log_det_cov = len(d) * np.log(scale) + np.log(np.linalg.det(cov / scale))

    return -chi2 - log_det_cov


def inf_smooth_lnlike(
    pars: Union[List[float], Tuple[float, ...]],
    data: Tuple[np.ndarray]
) -> float:
    # Check parameter priors
    bias_p, bias_s, eta_0, eta_m, eta_s, gamma_p, gamma_s, r_inf, mu, logd = pars
    out_of_prior_neg = any(
        [p < 0 for p in (bias_p, bias_s, eta_0, eta_s, gamma_p, gamma_s, r_inf, mu)])
    out_of_prior_lrg = any([p < 0 for p in (eta_0, eta_s)])
    out_of_prior_eta = (eta_m < -1) & (eta_m > 1)
    out_of_prior_logd = (-4 > logd) & (logd > 0)
    if out_of_prior_neg or out_of_prior_lrg or out_of_prior_eta or out_of_prior_logd:
        return -np.inf

    # Unpack data and evaluate parameters
    x, y, covy, mask, morb, xi_large_call, r_h, m_pivot = data
    delta = np.power(10., logd)
    bias = power_law(morb / m_pivot, p=bias_p, s=bias_s)
    eta = error_func_pos_decr(np.log10(morb / m_pivot), eta_0, eta_m, eta_s)
    gamma = power_law(morb / m_pivot, p=gamma_p, s=gamma_s)

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for n_bin in range(y.shape[0]):
        # Reduce dynamical range of y and covy by multiplying by x^2
        x2 = x[mask[n_bin]]**2
        # Compute model deviation from data.
        d = x2 * (y[n_bin, mask[n_bin]] - xi_inf_model(r=x[mask[n_bin]],
                                                       r_h=r_h[n_bin], bias=bias[n_bin],
                                                       xi_large=xi_large_call, eta=eta[n_bin],
                                                       gamma=gamma[n_bin], r_inf=r_inf, mu=mu))
        # Add regularization as a percent error to the covariance in quadrature.
        cov = np.outer(x2, x2) * (covy[n_bin, mask[n_bin], :][:, mask[n_bin]]
                                  + np.diag((delta * y[n_bin, mask[n_bin]])**2))

        # Compute chi2
        chi2 = np.dot(d, np.linalg.solve(cov, d))

        # Compute the log of the determinant of the covariance.
        scale = np.max(y[n_bin])
        log_det_cov = len(d) * np.log(scale) + \
            np.log(np.linalg.det(cov / scale))

        lnlike -= chi2 + log_det_cov

    return lnlike


def xihm_lnlike(
    pars: Union[List[float], Tuple[float, ...]],
    data: Tuple[np.ndarray]
) -> float:
    # Check parameter priors
    r_h, alpha, a, bias, eta, gamma, r_inf, mu, logd = pars
    out_of_prior_neg = any(
        [p < 0 for p in (r_h, alpha, a, eta, gamma, r_inf, mu)])
    out_of_prior_lrg = any([p > 10 for p in (bias, eta, gamma, r_inf, mu)])
    out_of_prior_bias = bias < 1
    out_of_prior_logd = (-4 > logd) & (logd > 0)
    if out_of_prior_neg or out_of_prior_lrg or out_of_prior_bias or out_of_prior_logd:
        return -np.inf

    # Unpack data
    x, y, covy, morb, xi_large_call, rhom = data
    delta = np.power(10., logd)

    # Reduce dynamical range of y and covy by multiplying by x^2
    x2 = x**2

    # Compute model deviation from data.
    d = x2 * (y - xihm_model(r=x, morb=morb, r_h=r_h, alpha=alpha, a=a,
                             bias=bias, xi_large=xi_large_call, eta=eta,
                             gamma=gamma, r_inf=r_inf, mu=mu, rhom=rhom))

    # Add regularization as a percent error to the covariance in quadrature.
    cov = np.outer(x2, x2) * (covy + np.diag((delta*y)**2))

    # Compute chi2
    chi2 = np.dot(d, np.linalg.solve(cov, d))

    # Compute the log of the determinant of the covariance.
    scale = np.max(y)
    log_det_cov = len(d) * np.log(scale) + np.log(np.linalg.det(cov / scale))

    return -chi2 - log_det_cov


def xihm_smooth_lnlike(
    pars: Union[List[float], Tuple[float, ...]],
    data: Tuple[np.ndarray]
) -> float:

    # Check parameter priors
    rh_p, rh_s, alpha_p, alpha_s, a, bias_p, bias_s, eta_0, eta_m, eta_s, gamma_p, gamma_s, r_inf, mu, logd = pars
    out_of_prior_neg = any([p < 0 for p in (
        rh_p, rh_s, alpha_p, a, bias_p, bias_s, eta_0, eta_s, gamma_p, gamma_s, r_inf, mu)])
    out_of_prior_pos = any([p > 0 for p in (alpha_s, )])
    out_of_prior_lrg = any([p < 0 for p in (eta_0, eta_s)])
    out_of_prior_eta = (eta_m < -1) & (eta_m > 1)
    out_of_prior_logd = (-4 > logd) & (logd > 0)
    if out_of_prior_neg or out_of_prior_pos or out_of_prior_lrg or out_of_prior_eta or out_of_prior_logd:
        return -np.inf

    # Unpack data and evaluate parameters
    x, y, covy, mask, morb, m_pivot, xi_large_call, rhom = data
    delta = np.power(10., logd)
    r_h = power_law(morb / m_pivot, p=rh_p, s=rh_s)
    alpha = power_law(morb / m_pivot, p=alpha_p, s=alpha_s)
    bias = power_law(morb / m_pivot, p=bias_p, s=bias_s)
    eta = error_func_pos_decr(morb / m_pivot, eta_0, eta_m, eta_s)
    gamma = power_law(morb / m_pivot, p=gamma_p, s=gamma_s)

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for n_bin in range(y.shape[0]):
        # Reduce dynamical range of y and covy by multiplying by x^2
        x2 = x[mask[n_bin]]**2
        # Compute model deviation from data.
        d = x2 * (y[n_bin, mask[n_bin]] - xihm_model(r=x[mask[n_bin]], morb=morb[n_bin],
                                                     r_h=r_h[n_bin], alpha=alpha[n_bin], a=a,
                                                     bias=bias[n_bin], xi_large=xi_large_call, eta=eta[n_bin],
                                                     gamma=gamma[n_bin], r_inf=r_inf, mu=mu, rhom=rhom))
        # Add regularization as a percent error to the covariance in quadrature.
        cov = np.outer(x2, x2) * (covy[n_bin, mask[n_bin], :][:, mask[n_bin]]
                                  + np.diag((delta * y[n_bin, mask[n_bin]])**2))

        # Compute chi2
        chi2 = np.dot(d, np.linalg.solve(cov, d))

        # Compute the log of the determinant of the covariance.
        scale = np.max(y[n_bin])
        log_det_cov = len(d) * np.log(scale) + \
            np.log(np.linalg.det(cov / scale))

        lnlike -= chi2 + log_det_cov
    return lnlike


if __name__ == '__main__':
    pass
