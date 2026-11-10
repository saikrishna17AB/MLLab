"""
ALL COMMON PROBABILITY DISTRIBUTIONS IN ONE FILE
Requires: pip install numpy scipy

Discrete distributions: pmf(), cdf(), sf()
Continuous distributions: pdf(), cdf()
Common functions: ppf(), rvs(), mean(), var(), std()
"""

import numpy as np
from scipy.stats import (
    bernoulli, binom, poisson, geom, nbinom, hypergeom, multinomial,
    uniform, norm, expon, gamma, beta, chi2, t, f
)

np.set_printoptions(precision=5, suppress=True)


def show_discrete(name, dist, x):
    """Display common calculations for a discrete distribution."""
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"PMF P(X={x}):", dist.pmf(x))
    print(f"CDF P(X<={x}):", dist.cdf(x))
    print(f"P(X>{x}):", dist.sf(x))
    print("95th percentile:", dist.ppf(0.95))
    print("Mean:", dist.mean())
    print("Variance:", dist.var())
    print("Standard deviation:", dist.std())
    print("Five random samples:", dist.rvs(size=5, random_state=42))


def show_continuous(name, dist, x, lower, upper):
    """Display common calculations for a continuous distribution."""
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"PDF at x={x}:", dist.pdf(x))
    print(f"CDF P(X<={x}):", dist.cdf(x))
    print(f"P({lower}<=X<={upper}):",
          dist.cdf(upper) - dist.cdf(lower))
    print("95th percentile:", dist.ppf(0.95))
    print("Mean:", dist.mean())
    print("Variance:", dist.var())
    print("Standard deviation:", dist.std())
    print("Five random samples:", dist.rvs(size=5, random_state=42))


# ============================================================
# A. DISCRETE DISTRIBUTIONS
# ============================================================

# 1. Bernoulli: one trial, success (1) or failure (0).
show_discrete("1. Bernoulli (p=0.6)", bernoulli(p=0.6), x=1)

# 2. Binomial: number of successes in n independent trials.
show_discrete("2. Binomial (n=10, p=0.5)", binom(n=10, p=0.5), x=5)

# 3. Poisson: number of events in a fixed interval; mu is lambda.
show_discrete("3. Poisson (lambda=4)", poisson(mu=4), x=3)

# 4. Geometric: trial number on which the first success occurs.
show_discrete("4. Geometric (p=0.3)", geom(p=0.3), x=4)

# 5. Negative binomial: number of failures before r successes.
show_discrete(
    "5. Negative Binomial (r=5, p=0.4)",
    nbinom(n=5, p=0.4), x=3
)

# 6. Hypergeometric: sampling without replacement.
# M=population size, n=successes in population, N=draws.
show_discrete(
    "6. Hypergeometric (M=20, n=7, N=5)",
    hypergeom(M=20, n=7, N=5), x=2
)

# 7. Multinomial: counts across three or more categories.
multi_n = 10
multi_p = np.array([0.2, 0.5, 0.3])
multi = multinomial(n=multi_n, p=multi_p)
observed_counts = np.array([2, 5, 3])

print("\n" + "=" * 60)
print("7. Multinomial (n=10, p=[0.2, 0.5, 0.3])")
print("=" * 60)
print("P(counts [2, 5, 3]):", multi.pmf(observed_counts))
print("Mean vector:", multi_n * multi_p)
print("Covariance matrix:")
print(multi_n * (np.diag(multi_p) - np.outer(multi_p, multi_p)))
print("Five random samples:")
print(multi.rvs(size=5, random_state=42))


# ============================================================
# B. CONTINUOUS DISTRIBUTIONS
# ============================================================

# 8. Uniform: constant density between a and b.
# SciPy: loc=a, scale=b-a.
show_continuous(
    "8. Uniform (a=2, b=10)",
    uniform(loc=2, scale=8), x=5, lower=4, upper=8
)

# 9. Normal/Gaussian: loc=mean, scale=standard deviation.
show_continuous(
    "9. Normal (mean=50, standard deviation=10)",
    norm(loc=50, scale=10), x=60, lower=40, upper=60
)

# 10. Exponential: SciPy scale = 1/lambda.
lam = 2
show_continuous(
    "10. Exponential (lambda=2)",
    expon(scale=1 / lam), x=1, lower=0.5, upper=1.5
)

# 11. Gamma: a=shape, scale=scale parameter.
show_continuous(
    "11. Gamma (shape=2, scale=3)",
    gamma(a=2, scale=3), x=5, lower=2, upper=8
)

# 12. Beta: values are between 0 and 1.
show_continuous(
    "12. Beta (a=2, b=5)",
    beta(a=2, b=5), x=0.4, lower=0.2, upper=0.6
)

# 13. Chi-square: df=degrees of freedom.
show_continuous(
    "13. Chi-square (df=5)",
    chi2(df=5), x=4, lower=2, upper=8
)

# 14. Student's t-distribution: df=degrees of freedom.
show_continuous(
    "14. Student's t (df=10)",
    t(df=10), x=1, lower=-1, upper=1
)

# 15. F-distribution: dfn and dfd are degrees of freedom.
show_continuous(
    "15. F-distribution (dfn=5, dfd=10)",
    f(dfn=5, dfd=10), x=2, lower=0.5, upper=3
)


# ============================================================
# C. EXTRA: NORMAL DISTRIBUTION INTERVAL PROBABILITY
# ============================================================

# P(40 <= X <= 60) = CDF(60) - CDF(40).
prob = norm.cdf(60, loc=50, scale=10) - norm.cdf(40, loc=50, scale=10)
print("\nP(40 <= X <= 60) for N(50, 10^2):", prob)

print("\nAll distribution examples completed.")
