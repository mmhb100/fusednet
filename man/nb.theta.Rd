\name{nb.theta}
\alias{nb.theta}
\title{Maximum-likelihood estimate of the negative binomial dispersion}
\description{Estimates \eqn{\theta} in the NB2 model (variance \eqn{\mu + \mu^2/\theta}) for given
means \eqn{\mu}, by Newton's method on \eqn{\log\theta}. Used internally by
\code{fusednet(family = "negbin", theta = NULL)}; gives the same result as \code{MASS::theta.ml}.}
\usage{nb.theta(y, mu, theta0 = NULL, maxit = 50, tol = 1e-08)}
\arguments{
  \item{y}{non-negative counts.}
  \item{mu}{fitted means.}
  \item{theta0}{starting value (default: moment estimate).}
  \item{maxit}{maximum Newton iterations.}
  \item{tol}{convergence tolerance on the log-scale step.}
}
\value{the estimate of \eqn{\theta} (bounded to \eqn{[10^{-4}, 10^8]}).}
\examples{
set.seed(1)
mu <- exp(rnorm(200, 1, 0.5)); y <- rnbinom(200, size = 3, mu = mu)
nb.theta(y, mu)
}
