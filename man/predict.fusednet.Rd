\name{predict.fusednet}
\alias{predict.fusednet}
\alias{coef.fusednet}
\alias{print.fusednet}
\alias{plot.fusednet}
\title{Predict, extract coefficients, print and plot a fusednet fit}
\description{Methods for \code{"fusednet"} objects, mirroring those in \pkg{glmnet}.}
\usage{
\method{predict}{fusednet}(object, newx, s = NULL,
        type = c("link", "response", "class", "coefficients", "nonzero"),
        exact = FALSE, ...)
\method{coef}{fusednet}(object, s = NULL, exact = FALSE, ...)
\method{print}{fusednet}(x, digits = max(3, getOption("digits") - 3), ...)
\method{plot}{fusednet}(x, xvar = c("lambda", "norm", "dev"),
     type = c("path", "profile"), s = NULL, label = FALSE, ...)
}
\arguments{
  \item{object, x}{a fitted \code{"fusednet"} object.}
  \item{newx}{matrix of new observations.}
  \item{s}{value(s) of \code{lambda} at which predictions are required. Default is the entire path.
    Values not on the path are linearly interpolated (as in glmnet) unless \code{exact = TRUE}.}
  \item{type}{\code{"link"} linear predictor; \code{"response"} fitted probability (binomial), fitted mean (poisson, negbin), relative risk \eqn{e^\eta} (cox) or fitted value;
    \code{"class"} predicted class; \code{"coefficients"}; \code{"nonzero"} indices of non-zero coefficients.}
  \item{exact}{if \code{TRUE} and \code{s} is not on the path, the model is refitted including \code{s}. Supply the original \code{x} and \code{y} via \code{...} if they are not available in the calling environment.}
  \item{xvar}{x-axis variable for coefficient paths.}
  \item{label}{label the curves with variable names.}
  \item{digits}{significant digits in printout.}
  \item{...}{further arguments (passed to \code{matplot} for plots).}
}
\details{\code{plot(fit, type = "profile", s = )} draws the coefficients against their index at a given lambda, which shows the fused (piecewise-constant) structure.}
\value{Predictions, coefficient matrices, or (invisibly) plotted values.}
