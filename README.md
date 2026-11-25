# fusednet: fast sparse fused lasso regression in C++ (glmnet-style)

`fusednet` fits **sparse fused lasso** linear, logistic, Poisson, negative binomial and
Cox regression over a full regularisation path, with a glmnet-like interface, using a
C++ core (Rcpp + RcppArmadillo).

$$
\min_{\beta_0,\beta}\; \frac{1}{2n}\lVert y-\beta_0-X\beta\rVert_2^2
\;+\;\lambda_1\sum_{j=1}^p|\beta_j|\;+\;\lambda_2\sum_{j=1}^{p-1}|\beta_{j+1}-\beta_j|
$$

For the other families the squared error is replaced by the negative log-likelihood divided by n
(the same scaling as glmnet):

| `family` | Response | Loss per observation |
|---|---|---|
| `"gaussian"` | numeric | (y - eta)^2 / 2 |
| `"binomial"` | 0/1, logical or 2-level factor | log(1 + e^eta) - y eta |
| `"poisson"` | counts | e^eta - y eta |
| `"negbin"` | counts | (y + theta) log(theta + e^eta) - y eta, theta fixed or estimated |
| `"cox"` | `Surv(time, status)` | Breslow partial likelihood (ties), no intercept |

[![R-CMD-check](https://github.com/mmhb100/fusednet/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/mmhb100/fusednet/actions/workflows/R-CMD-check.yaml)

## Installation

You need a C++ toolchain (Windows: **Rtools**; macOS: **Xcode command line tools**; Linux: `r-base-dev`).

```r
# install.packages("remotes")
remotes::install_github("mmhb100/fusednet")
```

## Quick start

```r
library(fusednet)
fit <- fusednet(x, y, alpha = 0.5)          # whole path: lambda1 = alpha*lambda, lambda2 = (1-alpha)*lambda
print(fit); plot(fit)                        # coefficient paths
plot(fit, type = "profile", s = fit$lambda[30])   # piecewise-constant coefficient profile

cv <- cv.fusednet(x, y, alpha = 0.5, nfolds = 10)  # cross-validation
plot(cv); coef(cv, s = "lambda.min"); predict(cv, newx, s = "lambda.1se")

fitb <- fusednet(x, yb, family = "binomial")       # logistic fused lasso
fitp <- fusednet(x, counts, family = "poisson")    # Poisson
fitn <- fusednet(x, counts, family = "negbin")     # negative binomial, theta estimated per lambda
fitn2 <- fusednet(x, counts, family = "negbin", theta = 2)   # ... or theta fixed
fitc <- fusednet(x, survival::Surv(time, status), family = "cox")   # Cox

cv.fusednet(x, survival::Surv(time, status), family = "cox", type.measure = "C")
```

### Translating your CVXR code

```r
# CVXR:
#   obj <- sum_squares(y - b0 - x %*% b)/(2*n) + lambda1*p_norm(b,1) + lambda2*p_norm(diff(b),1)
# fusednet (identical problem):
fit <- fusednet(x, y, lambda1 = lambda1, lambda2 = lambda2)
coef(fit)
```

If your CVXR objective uses `sum_squares(...)` **without** dividing by `2n`, divide your
penalties by `2n` to get the same solution: `lambda1 / (2*n)`, `lambda2 / (2*n)`.

Vectors of `lambda1`, `lambda2` are fitted as a warm-started sequence of pairs.

## Main functions

| Function | Purpose |
|---|---|
| `fusednet()` | fit the path (gaussian / binomial), `alpha` mixing, user `lambda1`/`lambda2`, `groups`, `standardize`, `intercept` |
| `cv.fusednet()` | K-fold CV, `lambda.min`, `lambda.1se`, measures mse/mae/deviance/class/auc/C (Harrell), parallel folds |
| `predict()`, `coef()` | glmnet-style, with `s =` interpolation, `type = "link"/"response"/"class"/"coefficients"/"nonzero"` |
| `plot()` | coefficient paths (`xvar = "lambda"/"norm"/"dev"`) or `type = "profile"` |
| `kkt.fusednet()` | exact KKT optimality certificate of a fit |
| `flsa()` | exact O(n) 1D fused lasso signal approximator |
| `relax.fusednet()` | relaxed (debiased) fit: keeps the selected fused blocks, re-estimates their heights without penalty |
| `nb.theta()` | ML estimate of the negative binomial dispersion (same as `MASS::theta.ml`) |

`groups` (contiguous labels) restricts fusion to adjacent coefficients within the same group,
e.g. several genomic chromosomes, or several time series stacked side by side.

## Algorithm

* **FISTA** (accelerated proximal gradient) with **adaptive restart** and a rounding-stable backtracking line search.
* The proximal operator of `lambda1*|b| + lambda2*TV(b)` is computed **exactly in O(p)**:
  Condat's direct 1D total-variation algorithm, followed by soft-thresholding (Friedman et al. 2007).
* **Warm starts** along a decreasing lambda path; **exact lambda_max** (smallest lambda giving beta = 0)
  computed by interval propagation of the subgradient conditions.
* **Covariance mode** (precomputed `X'X`, used automatically when n > p) or **naive mode** (n <= p),
  sparse matrix–vector products when many coefficients are zero.
* Stopping rule: proximal-gradient mapping below `thresh * lambda` (default `1e-7`), which bounds the KKT residual.

## Accuracy and speed (tested)

* Objective values match ECOS (the interior-point solver used by CVXR) to 9–10 digits and are always
  equal or lower; KKT residuals ≈ 1e-7 relative to lambda.
* `alpha = 1` reproduces `glmnet` lasso coefficients for gaussian, binomial, Poisson and Cox (`standardize = FALSE`),
  to about 1e-8. For the negative binomial, fusednet reaches KKT residuals of ~1e-7 where `glmnet` with
  `MASS::negative.binomial(theta)` reports non-convergence.
* With no penalty, fusednet reproduces `survival::coxph(ties = "breslow")` and `MASS::glm.nb` (coefficients and theta).

Full lambda path, alpha = 0.5 (reference BLAS, 2-core machine):

| n | p | # lambdas | fusednet | ECOS (CVXR backend), same path | speed-up |
|---|---|---|---|---|---|
| 500 | 100 | 75 | 0.01 s | 8 s | ~600x |
| 200 | 1000 | 97 | 2.3 s | 1288 s | ~560x |
| 5000 | 500 | 76 | 0.6 s | 2625 s | ~4200x |
| 1000 | 5000 | 94 | 65 s (24 s with OpenBLAS) | — | — |

CVXR adds its own problem-compilation overhead on top of the solver, so the real speed-up versus CVXR is larger.
See `inst/examples/compare_with_CVXR.R` to reproduce the comparison on your machine.

**Tip:** for large p > n problems, linking R to an optimised BLAS (OpenBLAS, MKL, Apple Accelerate)
speeds fusednet up several-fold, since its inner loop is two matrix–vector products.

## Notes

* Columns of `x` must be in their natural order (neighbours are fused).
* `standardize = FALSE` by default because the fusion penalty compares neighbouring coefficients,
  which is usually only meaningful on the original scale.
* Unlike glmnet's coordinate descent, the fused penalty is not separable, which is why a proximal method is used;
  for pure lasso, glmnet remains faster.

## References

* Tibshirani, Saunders, Rosset, Zhu & Knight (2005). Sparsity and smoothness via the fused lasso. *JRSS-B*.
* Friedman, Hastie, Hoefling & Tibshirani (2007). Pathwise coordinate optimization. *Ann. Appl. Stat.*
* Condat (2013). A direct algorithm for 1D total variation denoising. *IEEE SPL*.
* Beck & Teboulle (2009). FISTA. *SIAM J. Imaging Sciences*.
* O'Donoghue & Candès (2015). Adaptive restart for accelerated gradient schemes. *FoCM*.
