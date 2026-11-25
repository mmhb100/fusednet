#' KKT optimality check for a fusednet fit
#'
#' Returns, for every lambda on the path, the smallest perturbation eps of the
#' stationarity equations that makes them solvable, divided by lambda.
#' Values near 0 (e.g. < 1e-4) certify that the solution is (numerically) optimal.
#' @export
kkt.fusednet <- function(object, x, y, ztol = 1e-10) {
  x <- as.matrix(x); n <- nrow(x); p <- ncol(x)
  fam <- object$family
  py <- fn_prep_y(y, fam, n)
  center <- object$intercept || fam == "cox"
  xm <- if (center) colMeans(x) else rep(0, p)
  xs <- if (object$standardize) {
    xc <- x - rep(xm, each = n)
    s <- sqrt(colSums(xc^2) / n); s[s < .Machine$double.eps] <- 1; s
  } else rep(1, p)
  out <- numeric(length(object$lambda))
  for (l in seq_along(object$lambda)) {
    b <- object$beta[, l]
    eta <- object$a0[l] + drop(x %*% b)
    th <- if (fam == "negbin") object$theta[l] else 1
    r <- fn_eval(eta, py$y, fam, th, py$cx)$r
    g <- -drop(crossprod(x, r)) / n / xs   # negative gradient in fitting scale
    out[l] <- .kkt_violation(g, b * xs, object$lambda1[l], object$lambda2[l],
                             ztol, object$seg) / max(object$lambda[l], 1e-300)
  }
  out
}

#' Fused lasso signal approximator (exact, O(n))
#'
#' Solves min_b 0.5*||y - b||^2 + lambda1*||b||_1 + lambda2*sum|b_{i+1} - b_i|.
#' @export
flsa <- function(y, lambda1 = 0, lambda2 = 0, groups = NULL) {
  y <- as.numeric(y)
  seg <- fn_segments(groups, length(y))
  drop(.prox_sfl(y, lambda1, lambda2, seg))
}

#' Relaxed (debiased) fused lasso coefficients
#'
#' Keeps the sparsity and fusion pattern of a fit and re-estimates the value of
#' each non-zero fused block without penalty.
#' @export
relax.fusednet <- function(object, x, y, s = NULL, gamma = 0, maxit = 100000L) {
  if (gamma < 0 || gamma > 1) stop("'gamma' must be in [0, 1]")
  x <- as.matrix(x); p <- ncol(x)
  b <- coef(object, s = s)
  out <- b
  for (k in seq_len(ncol(b))) {
    bk <- b[-1, k]
    nz <- bk != 0
    if (!any(nz)) next
    newblock <- c(TRUE, diff(bk) != 0)
    newblock[object$seg + 1L] <- TRUE
    id <- cumsum(newblock)
    blocks <- unique(id[nz])
    Z <- vapply(blocks, function(g) rowSums(x[, id == g, drop = FALSE]), numeric(nrow(x)))
    Z <- matrix(Z, nrow = nrow(x))
    rf <- fusednet(Z, y, family = object$family, lambda1 = 0, lambda2 = 0,
                   intercept = object$intercept, thresh = 1e-10, maxit = maxit)
    gam <- rf$beta[, 1]
    refit <- numeric(p)
    for (g in seq_along(blocks)) refit[id == blocks[g]] <- gam[g]
    out[, k] <- gamma * b[, k] + (1 - gamma) * c(rf$a0[1], refit)
  }
  out
}
