import numpy as np
from scipy import stats

def calculate_statistical_error(n_tracks, n_collisions, confidence_level=0.6827):
    ### Calculates statistical errors for the ratio N_T / N_BX.
    ## Returns the rate, the approximate symmetric error, and the exact asymmetric errors.

    rate = n_tracks / n_collisions
    
    # 1. Standard Approximation (Gaussian limit): sqrt(N) / N_BX
    error_approx = np.sqrt(n_tracks) / n_collisions
    
    # 2. Exact Poisson Confidence Interval (Garwood Interval) for the small number limit (w/asymmetric errors), using the chi2 distribution relationship to Poisson limits.
    alpha = 1.0 - confidence_level
    
    # Lower Bound
    if n_tracks == 0:
        mu_low = 0.0
    else:
        # Lower limit for mean of Poisson
        mu_low = stats.chi2.ppf(alpha / 2, 2 * n_tracks) / 2.0
        
    # Upper Bound
    # Upper limit for mean of Poisson
    mu_high = stats.chi2.ppf(1.0 - alpha / 2, 2 * (n_tracks + 1)) / 2.0
    
    # Convert count limits to rate limits
    rate_low = mu_low / n_collisions
    rate_high = mu_high / n_collisions
    
    # Calculate error bars (distance from central value)
    error_minus = rate - rate_low
    error_plus = rate_high - rate
    
    return {
        "rate": rate,
        "symmetric_error": error_approx,
        "asymmetric_error_plus": error_plus,
        "asymmetric_error_minus": error_minus
    }





# signal
res1 = calculate_statistical_error(422, 3655)
print(f"Case 1 (422/3655):")
print(f"Rate: {res1['rate']:.4f}")
print(f"Error (Approx): +/- {res1['symmetric_error']:.4f}")
print(f"Error (Exact):  +{res1['asymmetric_error_plus']:.4f} / -{res1['asymmetric_error_minus']:.4f}")
print("-" * 30)

# background
res2 = calculate_statistical_error(5, 205907)
print(f"Case 2 (5/205907):")
print(f"Rate: {res2['rate']:.2e}")
print(f"Error (Approx): +/- {res2['symmetric_error']:.2e}")
print(f"Error (Exact):  +{res2['asymmetric_error_plus']:.2e} / -{res2['asymmetric_error_minus']:.2e}")
