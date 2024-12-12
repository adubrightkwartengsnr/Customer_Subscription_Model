import numpy as np
def safe_log1p(X):
    X = np.where(X <= 0, 1e-9, X)  # Replace non-positive values with a small positive constant
    return np.log1p(X)
# Add `safe_log1p` to global namespace to resolve deserialization issues
globals()['safe_log1p'] = safe_log1p