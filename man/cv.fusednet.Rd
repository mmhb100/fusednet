\name{cv.fusednet}
\alias{cv.fusednet}
\alias{coef.cv.fusednet}
\alias{predict.cv.fusednet}
\alias{print.cv.fusednet}
\alias{plot.cv.fusednet}
\title{Cross-validation for fusednet}
\description{K-fold cross-validation over the lambda path (for a fixed \code{alpha}), returning \code{lambda.min} and \code{lambda.1se} like \code{cv.glmnet}.}
\usage{
cv.fusednet(x, y, ..., nfolds = 10, foldid = NULL,
            type.measure = c("default", "mse", "mae", "deviance", "class", "auc", "C"),
            ncores = 1L, keep = FALSE)
\method{coef}{cv.fusednet}(object, s = c("lambda.1se", "lambda.min"), ...)
\method{predict}{cv.fusednet}(object, newx, s = c("lambda.1se", "lambda.min"), ...)
\method{print}{cv.fusednet}(x, digits = max(3, getOption("digits") - 3), ...)
\method{plot}{cv.fusednet}(x, sign.lambda = 1, ...)
}
\arguments{
  \item{x, y}{data, as in \code{\link{fusednet}}.}
  \item{...}{arguments passed to \code{\link{fusednet}} (e.g. \code{family}, \code{alpha}, \code{lambda1}/\code{lambda2}); for methods, passed on to \code{predict.fusednet}.}
  \item{nfolds}{number of folds.}
  \item{foldid}{optional vector of fold ids (use the same \code{foldid} to compare different \code{alpha} values fairly).}
  \item{type.measure}{loss: \code{"mse"} (default for gaussian), \code{"deviance"} (default for the other families),
    \code{"mae"}, \code{"class"} and \code{"auc"} (binomial), \code{"C"} (Harrell's concordance, cox).
    For cox, \code{"deviance"} is the cross-validated partial likelihood of Verweij and van Houwelingen (1993),
    per held-out event. For negbin, \code{"deviance"} is computed as \eqn{-2} times the full held-out
    log-likelihood, which (unlike the deviance) is comparable when \eqn{\theta} is re-estimated at each lambda.}
  \item{ncores}{number of cores used for folds (via \code{parallel::mclapply}; ignored on Windows).}
  \item{keep}{return the pre-validated fits?}
  \item{object}{a \code{"cv.fusednet"} object.}
  \item{newx}{new data.}
  \item{s}{\code{"lambda.1se"}, \code{"lambda.min"}, or numeric lambda values.}
  \item{sign.lambda}{plot against \code{log(lambda)} (1) or \code{-log(lambda)} (-1).}
  \item{digits}{significant digits.}
}
\value{An object of class \code{"cv.fusednet"} with \code{lambda}, \code{cvm}, \code{cvsd}, \code{cvup}, \code{cvlo}, \code{nzero}, \code{nfused}, \code{lambda.min}, \code{lambda.1se}, \code{fusednet.fit} and \code{foldid}.}
\examples{
set.seed(1)
x <- matrix(rnorm(100 * 40), 100)
y <- drop(x[, 11:20] \%*\% rep(1, 10) + rnorm(100))
cv <- cv.fusednet(x, y, alpha = 0.5, nfolds = 5)
plot(cv)
coef(cv, s = "lambda.min")

# tune alpha with common folds
foldid <- sample(rep(1:5, length.out = 100))
cvs <- lapply(c(0.1, 0.5, 0.9), function(a) cv.fusednet(x, y, alpha = a, foldid = foldid))
sapply(cvs, function(cv) min(cv$cvm))
}
