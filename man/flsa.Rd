\name{flsa}
\alias{flsa}
\title{Fused lasso signal approximator}
\description{Exact \eqn{O(n)} solution of
\eqn{\min_b \frac12\|y-b\|^2 + \lambda_1\|b\|_1 + \lambda_2\sum_i|b_{i+1}-b_i|}{min_b 0.5||y - b||^2 + lambda1 ||b||_1 + lambda2 sum|b_{i+1} - b_i|}
(1D sparse fused lasso denoising), the proximal operator used inside \code{fusednet}.}
\usage{flsa(y, lambda1 = 0, lambda2 = 0, groups = NULL)}
\arguments{
  \item{y}{numeric signal.}
  \item{lambda1}{sparsity penalty.}
  \item{lambda2}{fusion (total variation) penalty.}
  \item{groups}{optional contiguous labels; fusion is applied only within groups.}
}
\value{the denoised signal.}
\examples{
set.seed(1)
y <- c(rep(0, 50), rep(3, 30), rep(-2, 20)) + rnorm(100)
plot(y); lines(flsa(y, lambda1 = 0.1, lambda2 = 5), col = "red", lwd = 2)
}
