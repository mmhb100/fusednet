\name{relax.fusednet}
\alias{relax.fusednet}
\title{Relaxed (debiased) fused lasso coefficients}
\description{
The lasso and fused lasso penalties shrink coefficients towards zero, so the
estimated block heights are biased downwards. \code{relax.fusednet} keeps the
sparsity and fusion pattern selected at \code{s} (which coefficients are zero,
which neighbours are equal) and re-estimates the value of each non-zero block
by unpenalised maximum likelihood (the "relaxed lasso" of Meinshausen, 2007,
applied to fused blocks). Use \code{gamma} to mix the two fits.
}
\usage{relax.fusednet(object, x, y, s = NULL, gamma = 0, maxit = 100000L)}
\arguments{
  \item{object}{a \code{"fusednet"} fit (e.g. \code{cv$fusednet.fit}).}
  \item{x, y}{the data used for fitting.}
  \item{s}{lambda value(s); default the whole path.}
  \item{gamma}{mixing: \code{0} = fully relaxed (unpenalised block values),
    \code{1} = the original penalised fit; in between, a convex combination.}
  \item{maxit}{maximum iterations for each refit.}
}
\details{Relaxing removes the shrinkage bias but increases variance, and works best
when the selected block structure is right and each block has enough data; choose
\code{s} and \code{gamma} by cross-validation or held-out data. For binomial data
that are (nearly) separable on the selected blocks, the unpenalised refit may
diverge; use \code{gamma > 0} in that case.}
\value{coefficient matrix (intercept first), one column per value of \code{s}.}
\references{Meinshausen, N. (2007). Relaxed lasso. \emph{Computational Statistics & Data Analysis}, 52, 374--393.}
\examples{
set.seed(1)
x <- matrix(rnorm(200 * 50), 200)
b <- c(rep(0, 10), rep(1, 10), rep(0, 30))
y <- drop(x \%*\% b + rnorm(200))
cv <- cv.fusednet(x, y, nfolds = 5)
b_lasso <- coef(cv, s = "lambda.min")
b_relax <- relax.fusednet(cv$fusednet.fit, x, y, s = cv$lambda.min)
plot(b, type = "s", lty = 2)
lines(b_lasso[-1], type = "s", col = 4)
lines(b_relax[-1], type = "s", col = 2)
}
