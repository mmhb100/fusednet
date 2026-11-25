\name{fusednet}
\alias{fusednet}
\title{Fit a sparse fused lasso regression path (linear, logistic, Poisson, negative binomial, Cox)}
\description{
Fits linear, logistic, Poisson, negative binomial or Cox regression with the sparse fused lasso penalty
\eqn{\lambda_1\sum_j|\beta_j| + \lambda_2\sum_j|\beta_{j+1}-\beta_j|} over a decreasing
sequence of \eqn{\lambda} values, using warm starts, like \code{glmnet}.
}
\usage{
fusednet(x, y, family = c("gaussian", "binomial", "poisson", "negbin", "cox"),
         alpha = 0.5, nlambda = 100,
         lambda.min.ratio = ifelse(nrow(x) < ncol(x), 0.01, 1e-04),
         lambda = NULL, lambda1 = NULL, lambda2 = NULL,
         groups = NULL, intercept = TRUE, standardize = FALSE,
         theta = NULL, thresh = 1e-07, maxit = 100000L,
         type.solver = c("auto", "covariance", "naive"), early.stop = TRUE)
}
\arguments{
  \item{x}{numeric matrix, \code{n x p}. Columns must be in their natural (fusion) order.}
  \item{y}{response. Numeric for \code{"gaussian"}; 0/1, logical, or a two-level factor for \code{"binomial"};
    non-negative counts for \code{"poisson"} and \code{"negbin"}; a \code{survival::Surv(time, status)} object
    or a two-column matrix (time, status) for \code{"cox"}.}
  \item{family}{\code{"gaussian"} (least squares), \code{"binomial"} (logistic), \code{"poisson"} (log link),
    \code{"negbin"} (negative binomial NB2, log link) or \code{"cox"} (proportional hazards, Breslow method for ties).}
  \item{alpha}{mixing parameter in \eqn{[0,1]}: \eqn{\lambda_1 = \alpha\lambda} (sparsity) and
    \eqn{\lambda_2 = (1-\alpha)\lambda} (fusion). \code{alpha = 1} is the lasso, \code{alpha = 0} pure fusion (total variation).}
  \item{nlambda}{number of \eqn{\lambda} values.}
  \item{lambda.min.ratio}{smallest \eqn{\lambda} as a fraction of \eqn{\lambda_{max}} (the exact smallest value giving the null model).}
  \item{lambda}{optional user-supplied decreasing \eqn{\lambda} sequence (used with \code{alpha}).}
  \item{lambda1, lambda2}{optional user-supplied penalty values, used directly instead of \code{lambda}/\code{alpha}
    (recycled to a common length; pairs are fitted in order of decreasing \code{lambda1 + lambda2}). Use these to reproduce a CVXR problem exactly.}
  \item{groups}{optional vector of length \code{p} with contiguous labels; fusion is only applied between adjacent columns with the same label (several independent chains).}
  \item{intercept}{fit an (unpenalised) intercept? Ignored (no intercept) for \code{"cox"}.}
  \item{theta}{negative binomial dispersion (variance = mu + mu^2/theta). If \code{NULL} (default) it is
    estimated by maximum likelihood at every lambda, alternating with the penalised fit (like \code{MASS::glm.nb});
    a positive number keeps it fixed. Only for \code{"negbin"}.}
  \item{standardize}{standardise columns of \code{x} before fitting? Coefficients are always returned on the original scale. Default \code{FALSE} because the fusion penalty compares neighbouring coefficients, which is usually meaningful only on the original scale.}
  \item{thresh}{convergence tolerance: iterations stop when the proximal-gradient mapping (a bound on the KKT residual) is below \code{thresh * lambda}.}
  \item{maxit}{maximum number of iterations per \eqn{\lambda}.}
  \item{type.solver}{\code{"covariance"} precomputes \eqn{X^TX} (fast when \eqn{n > p}); \code{"naive"} works with \eqn{X} directly (fast when \eqn{p > n}); \code{"auto"} chooses.}
  \item{early.stop}{stop the path early (like glmnet) when the deviance explained exceeds 99.9\% or stops changing.}
}
\details{
The smooth part of the objective is the negative log-likelihood divided by \eqn{n}:
squared error / 2 (gaussian); \eqn{\log(1+e^\eta)-y\eta} (binomial); \eqn{e^\eta - y\eta} (poisson);
\eqn{(y+\theta)\log(\theta+e^\eta) - y\eta} (negbin); minus the Breslow log partial likelihood (cox).
This matches \code{glmnet}'s scaling, so \code{alpha = 1} reproduces \code{glmnet}'s lasso with
\code{standardize = FALSE}.
}
\value{An object of class \code{"fusednet"} with components
\item{a0}{intercepts} \item{beta}{\code{p x nlambda} coefficient matrix}
\item{lambda, lambda1, lambda2, alpha}{penalty values actually used}
\item{theta}{negative binomial dispersion at each lambda (\code{"negbin"} only)}
\item{df}{number of non-zero coefficients} \item{nfused}{number of distinct non-zero fused blocks (degrees of freedom of the fused lasso)}
\item{dev.ratio}{fraction of (null) deviance explained} \item{nulldev}{null deviance}
\item{iter}{iterations per lambda} \item{call, family, classnames, ...}{other details}
}
\seealso{\code{\link{cv.fusednet}}, \code{\link{predict.fusednet}}, \code{\link{kkt.fusednet}}}
\examples{
set.seed(1)
n <- 100; p <- 50
x <- matrix(rnorm(n * p), n)
beta <- c(rep(0, 10), rep(2, 10), rep(0, 10), rep(-1, 10), rep(0, 10))
y <- drop(x \%*\% beta + rnorm(n))
fit <- fusednet(x, y, alpha = 0.5)
print(fit)
plot(fit)
plot(fit, type = "profile", s = fit$lambda[30])

# the exact problem you would write in CVXR:
#   minimize( sum_squares(y - b0 - x \%*\% b)/(2*n) + l1*p_norm(b,1) + l2*p_norm(diff(b),1) )
fit2 <- fusednet(x, y, lambda1 = 0.05, lambda2 = 0.2)
coef(fit2)

# logistic regression
yb <- rbinom(n, 1, plogis(x \%*\% beta / 4))
fitb <- fusednet(x, yb, family = "binomial")
predict(fitb, x[1:5, ], s = fitb$lambda[20], type = "response")

# Poisson and negative binomial regression for counts
yc <- rpois(n, exp(x \%*\% beta / 4))
fitp <- fusednet(x, yc, family = "poisson")
fitnb <- fusednet(x, yc, family = "negbin")        # theta estimated at each lambda
fitnb$theta[1:5]

# Cox regression
if (requireNamespace("survival", quietly = TRUE)) {
  tt <- rexp(n, exp(x \%*\% beta / 4)); cc <- rexp(n, 0.2)
  ys <- survival::Surv(pmin(tt, cc), as.numeric(tt <= cc))
  fitc <- fusednet(x, ys, family = "cox")
  plot(fitc, type = "profile", s = fitc$lambda[30])
}
}
