\name{kkt.fusednet}
\alias{kkt.fusednet}
\title{KKT optimality certificate for a fusednet fit}
\description{For each lambda, computes the smallest perturbation of the subgradient
optimality conditions that makes them solvable (computed exactly in \eqn{O(p)} by interval
propagation), divided by lambda. Values close to zero certify the solution is optimal.}
\usage{kkt.fusednet(object, x, y, ztol = 1e-10)}
\arguments{
  \item{object}{a \code{"fusednet"} fit.}
  \item{x, y}{the data used for fitting.}
  \item{ztol}{tolerance for treating a coefficient (or a difference) as zero.}
}
\value{numeric vector of relative KKT violations, one per lambda.}
\examples{
x <- matrix(rnorm(200), 20); y <- rnorm(20)
fit <- fusednet(x, y)
max(kkt.fusednet(fit, x, y))
}
