\name{fusednet-package}
\alias{fusednet-package}
\docType{package}
\title{Fast Sparse Fused Lasso Regression Paths}
\description{
Fits sparse fused lasso penalised linear, logistic, Poisson, negative binomial and Cox regression over a path of
regularisation parameters, in the style of \pkg{glmnet}. The objective is
\deqn{\frac{1}{2n}\|y - \beta_0 - X\beta\|^2 + \lambda_1\sum_j|\beta_j| + \lambda_2\sum_{j}|\beta_{j+1}-\beta_j|}{(1/2n)||y - b0 - X b||^2 + lambda1 sum|b_j| + lambda2 sum|b_{j+1} - b_j|}
(or the corresponding negative mean log-likelihood, or Cox partial likelihood, for the other families).
The C++ solver uses FISTA with adaptive restart, where the proximal operator of the
penalty is computed exactly in \eqn{O(p)} via Condat's direct total-variation algorithm.
}
\seealso{\code{\link{fusednet}}, \code{\link{cv.fusednet}}, \code{\link{flsa}}, \code{\link{kkt.fusednet}}}
\references{
Tibshirani, R., Saunders, M., Rosset, S., Zhu, J. and Knight, K. (2005). Sparsity and smoothness via the fused lasso. \emph{JRSS B}, 67, 91--108.

Friedman, J., Hastie, T., Hoefling, H. and Tibshirani, R. (2007). Pathwise coordinate optimization. \emph{Ann. Appl. Stat.}, 1, 302--332.

Condat, L. (2013). A direct algorithm for 1D total variation denoising. \emph{IEEE Signal Proc. Letters}, 20, 1054--1057.

Beck, A. and Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. \emph{SIAM J. Imaging Sci.}, 2, 183--202.
}
\keyword{package}
