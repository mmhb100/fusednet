#' @useDynLib fusednet, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

FN_FAMILIES <- c("gaussian", "binomial", "poisson", "negbin", "cox")
fn_code <- function(family) match(family, FN_FAMILIES) - 1L

fn_segments <- function(groups, p) {
  if (is.null(groups)) return(0L)
  if (length(groups) != p) stop("'groups' must have length ncol(x)")
  r <- rle(as.vector(groups))
  if (anyDuplicated(r$values))
    stop("'groups' must be contiguous: columns of the same group must be adjacent")
  as.integer(c(0L, cumsum(r$lengths)[-length(r$lengths)]))
}

# Risk-set bookkeeping for the Cox partial likelihood (Breslow ties):
# ord = 0-based order of ascending time; gstart/gend = first/last sorted
# position (0-based) of each position's tie group.
fn_cox_struct <- function(time) {
  n <- length(time)
  ord <- order(time)
  ts <- time[ord]
  list(ord = as.integer(ord - 1L),
       gstart = as.integer(match(ts, ts) - 1L),
       gend = as.integer(n - match(ts, rev(ts))))
}
fn_empty_cx <- list(ord = integer(0), gstart = integer(0), gend = integer(0))

# Parse the response for each family. Returns the numeric vector passed to C++
# (the status indicator for cox) plus family-specific extras.
fn_prep_y <- function(y, family, n) {
  out <- list(classnames = NULL, time = NULL, cx = fn_empty_cx)
  if (family == "cox") {
    if (!(is.matrix(y) && ncol(y) == 2))
      stop("for family = \"cox\", y must be a survival::Surv object or a two-column matrix (time, status)")
    time <- as.numeric(y[, 1]); status <- as.numeric(y[, 2])
    if (any(time <= 0)) stop("survival times must be positive")
    if (!all(status %in% c(0, 1))) stop("status must be 0 (censored) or 1 (event)")
    if (sum(status) == 0) stop("there are no events")
    out$y <- status; out$time <- time; out$cx <- fn_cox_struct(time)
  } else if (family == "binomial") {
    if (is.factor(y) || is.character(y) || is.logical(y)) {
      y <- as.factor(y)
      if (nlevels(y) != 2) stop("binomial family requires a 2-level response")
      out$classnames <- levels(y)
      out$y <- as.numeric(y == out$classnames[2])
    } else {
      y <- as.numeric(y)
      if (!all(y %in% c(0, 1))) stop("binomial response must be 0/1, logical or a 2-level factor")
      out$classnames <- c("0", "1"); out$y <- y
    }
  } else {
    out$y <- as.numeric(y)
    if (family %in% c("poisson", "negbin") && any(out$y < 0))
      stop("counts must be non-negative for family = \"", family, "\"")
  }
  if (length(out$y) != n) stop("the number of responses must equal nrow(x)")
  out
}

# loss (mean over observations) and r = d(n * loss)/d eta at a linear predictor
fn_eval <- function(eta, y, family, theta = 1, cx = fn_empty_cx) {
  .family_eval(as.numeric(eta), as.numeric(y), fn_code(family),
               if (is.null(theta)) 1 else theta, cx$ord, cx$gstart, cx$gend)
}

# loss of the saturated model, so that deviance = 2 * n * (loss - fsat)
fn_fsat <- function(y, family, theta = 1, time = NULL) {
  ylogy <- ifelse(y > 0, y * log(pmax(y, 1e-300)), 0)
  switch(family,
         gaussian = 0, binomial = 0,
         poisson = mean(y - ylogy),
         negbin = mean((y + theta) * log(y + theta) - ylogy),
         cox = {
           d <- table(time[y == 1])
           sum(d * log(d)) / length(y)
         })
}

#' Maximum-likelihood estimate of the negative binomial dispersion theta
#'
#' @export
nb.theta <- function(y, mu, theta0 = NULL, maxit = 50, tol = 1e-8) {
  n <- length(y)
  if (is.null(theta0)) {
    v <- sum((y / mu - 1)^2)
    theta0 <- if (v > 0) n / v else 1e6
  }
  th <- min(max(theta0, 1e-4), 1e8)
  for (it in seq_len(maxit)) {
    score <- sum(digamma(y + th) - digamma(th) + log(th) + 1 - log(th + mu) - (y + th) / (th + mu))
    info  <- sum(trigamma(th) - trigamma(y + th) - 1 / th + 2 / (th + mu) - (y + th) / (th + mu)^2)
    step <- if (is.finite(info) && info > 0) score / (th * info) else sign(score) * 0.5
    step <- max(min(step, 2), -2)                    # Newton step on log(theta), damped
    th <- min(max(th * exp(step), 1e-4), 1e8)
    if (abs(step) < tol) break
  }
  th
}

# Null model (beta = 0): linear predictor, loss, deviance.
fn_null <- function(y, family, intercept, theta, cx, time) {
  n <- length(y)
  eta0 <- switch(family,
                 gaussian = if (intercept) mean(y) else 0,
                 binomial = if (intercept) stats::qlogis(min(max(mean(y), 1e-10), 1 - 1e-10)) else 0,
                 poisson = , negbin = if (intercept) log(max(mean(y), 1e-10)) else 0,
                 cox = 0)
  eta0 <- rep(eta0, n)
  ev <- fn_eval(eta0, y, family, theta, cx)
  fsat <- fn_fsat(y, family, theta, time)
  list(eta = eta0, r = ev$r, f = ev$loss, fsat = fsat,
       nulldev = 2 * n * (ev$loss - fsat))
}

fn_lambda_max <- function(xc, y, family, intercept, alpha, seg, theta, cx, nul) {
  n <- nrow(xc); p <- ncol(xc)
  if (alpha > 0) {
    g <- -drop(crossprod(xc, nul$r)) / n
    return(.lambda_max(g, alpha, seg))
  }
  # alpha == 0 (pure fusion): the null solution is constant within each segment
  starts <- c(seg, p) + 1L
  S <- vapply(seq_along(seg), function(s) {
    idx <- starts[s]:(starts[s + 1] - 1L)
    rowSums(xc[, idx, drop = FALSE])
  }, numeric(n))
  S <- matrix(S, nrow = n)
  icpt <- intercept && family != "cox"
  D <- if (icpt) cbind(1, S) else S
  start <- numeric(ncol(D)); if (icpt) start[1] <- nul$eta[1]
  fobj <- function(cf) fn_eval(drop(D %*% cf), y, family, theta, cx)$loss
  gobj <- function(cf) drop(crossprod(D, fn_eval(drop(D %*% cf), y, family, theta, cx)$r)) / n
  opt <- stats::optim(start, fobj, gobj, method = "BFGS",
                      control = list(maxit = 500, reltol = 1e-14))
  r <- fn_eval(drop(D %*% opt$par), y, family, theta, cx)$r
  g <- -drop(crossprod(xc, r)) / n
  lm <- 0
  for (s in seq_along(seg)) {
    idx <- starts[s]:(starts[s + 1] - 1L)
    lm <- max(lm, abs(cumsum(g[idx])))
  }
  lm
}

#' Fit a sparse fused lasso regression path
#'
#' See ?fusednet for details.
#' @export
fusednet <- function(x, y, family = c("gaussian", "binomial", "poisson", "negbin", "cox"),
                     alpha = 0.5, nlambda = 100,
                     lambda.min.ratio = ifelse(nrow(x) < ncol(x), 0.01, 1e-4),
                     lambda = NULL, lambda1 = NULL, lambda2 = NULL,
                     groups = NULL, intercept = TRUE, standardize = FALSE,
                     theta = NULL, thresh = 1e-7, maxit = 100000L,
                     type.solver = c("auto", "covariance", "naive"),
                     early.stop = TRUE) {
  this.call <- match.call()
  family <- match.arg(family)
  type.solver <- match.arg(type.solver)
  x <- as.matrix(x)
  if (!is.numeric(x)) stop("'x' must be a numeric matrix")
  storage.mode(x) <- "double"
  n <- nrow(x); p <- ncol(x)
  if (anyNA(x) || anyNA(y)) stop("missing values in x or y are not allowed")
  if (alpha < 0 || alpha > 1) stop("'alpha' must be in [0, 1]")
  if (family == "cox") intercept <- FALSE
  if (!is.null(theta) && (family != "negbin" || length(theta) != 1 || theta <= 0))
    stop("'theta' must be a single positive number, and only for family = \"negbin\"")
  vnames <- colnames(x)
  if (is.null(vnames)) vnames <- paste0("V", seq_len(p))

  py <- fn_prep_y(y, family, n)
  y <- py$y; cx <- py$cx
  seg <- fn_segments(groups, p)

  # ---- centring / scaling (cox: centring is harmless, partial likelihood is shift-invariant)
  center <- intercept || family == "cox"
  xm <- if (center) colMeans(x) else rep(0, p)
  xc <- if (center) x - rep(xm, each = n) else x
  if (standardize) {
    xs <- sqrt(colSums(xc^2) / n)
    xs[xs < .Machine$double.eps] <- 1
    xc <- xc / rep(xs, each = n)
  } else {
    xs <- rep(1, p)
  }
  ym <- if (family == "gaussian" && intercept) mean(y) else 0
  yfit <- if (family == "gaussian") y - ym else y

  estimate_theta <- family == "negbin" && is.null(theta)
  theta0 <- if (family != "negbin") 1
            else if (estimate_theta) nb.theta(y, rep(if (intercept) mean(y) else 1, n))
            else theta
  nul <- fn_null(y, family, intercept, theta0, cx, py$time)

  # ---- lambda sequence ----
  user_pairs <- !is.null(lambda1) || !is.null(lambda2)
  if (user_pairs) {
    if (is.null(lambda1)) lambda1 <- 0
    if (is.null(lambda2)) lambda2 <- 0
    len <- max(length(lambda1), length(lambda2))
    lambda1 <- rep_len(as.numeric(lambda1), len)
    lambda2 <- rep_len(as.numeric(lambda2), len)
    if (any(lambda1 < 0) || any(lambda2 < 0)) stop("lambda1 and lambda2 must be >= 0")
    tot <- lambda1 + lambda2
    o <- order(tot, decreasing = TRUE)
    if (!identical(o, seq_len(len))) {
      warning("lambda1/lambda2 pairs re-ordered by decreasing lambda1 + lambda2")
      lambda1 <- lambda1[o]; lambda2 <- lambda2[o]; tot <- tot[o]
    }
    lambda <- tot
    alphas <- ifelse(tot > 0, lambda1 / tot, alpha)
    early.stop <- FALSE
  } else {
    if (is.null(lambda)) {
      lmax <- fn_lambda_max(xc, y, family, intercept, alpha, seg, theta0, cx, nul)
      if (!is.finite(lmax) || lmax <= 0) lmax <- 1e-6
      lambda <- exp(seq(log(lmax), log(lmax * lambda.min.ratio), length.out = nlambda))
    } else {
      lambda <- sort(as.numeric(lambda), decreasing = TRUE)
      if (any(lambda < 0)) stop("lambda must be >= 0")
      early.stop <- FALSE
    }
    lambda1 <- alpha * lambda
    lambda2 <- (1 - alpha) * lambda
    alphas <- rep(alpha, length(lambda))
  }

  covmode <- switch(type.solver,
                    auto = family == "gaussian" && n > p && p <= 3000,
                    covariance = family == "gaussian",
                    naive = FALSE)
  icpt_cpp <- intercept && family %in% c("binomial", "poisson", "negbin")
  a0_init <- if (icpt_cpp) nul$eta[1] else 0

  # weights for the initial step-size estimate (largest curvature of the loss)
  # (curvature of the loss at the starting point; backtracking corrects it)
  wfun <- function(th, eta) switch(family,
    gaussian = rep(1, n), binomial = rep(0.25, n),
    poisson = rep(max(mean(y), 1e-2), n),
    negbin = { mu <- exp(eta); pmax((y + th) * th * mu / (th + mu)^2, 1e-3) },
    cox = pmax(nul$r + y, 1e-3))          # Nelson-Aalen cumulative hazard at eta = 0

  call_path <- function(l1, l2, th, b_init, a_init, nulldev, fsat, es) {
    eta_start <- if (length(b_init)) a_init + drop(xc %*% b_init) else rep(a_init, n)
    .fusednet_path(xc, yfit, fn_code(family), l1, l2,
                   icpt = icpt_cpp, thresh = thresh, maxit = as.integer(maxit),
                   covmode = covmode, nulldev = nulldev, fsat = fsat, seg = seg,
                   early_stop = es, a0_init = a_init, beta_init = b_init,
                   wL = wfun(th, eta_start), theta = th,
                   ord = cx$ord, gstart = cx$gstart, gend = cx$gend)
  }

  if (!estimate_theta) {
    fit <- call_path(lambda1, lambda2, theta0, numeric(0), a0_init,
                     nul$nulldev, nul$fsat, early.stop)
    nfit <- fit$nfit
    thetas <- rep(if (family == "negbin") theta0 else NA_real_, nfit)
    devs <- fit$dev; nulldevs <- rep(nul$nulldev, nfit)
    bfit <- fit$beta; afit <- drop(fit$a0); iter <- as.integer(fit$iter)
  } else {
    # negative binomial with unknown theta: alternate between the penalised fit
    # (warm-started) and the ML estimate of theta, at every lambda
    nl <- length(lambda)
    bfit <- matrix(0, p, nl); afit <- numeric(nl); iter <- integer(nl)
    thetas <- devs <- nulldevs <- numeric(nl)
    b <- numeric(p); a <- a0_init; th <- theta0; nfit <- 0
    for (l in seq_len(nl)) {
      it_tot <- 0L
      for (k in 1:50) {
        th_fit <- th                       # beta below is optimal for this theta
        fl <- call_path(lambda1[l], lambda2[l], th_fit, b, a, 0, 0, FALSE)
        b <- drop(fl$beta); a <- drop(fl$a0); it_tot <- it_tot + as.integer(fl$iter)
        mu <- exp(a + drop(xc %*% b))
        th <- nb.theta(y, mu, th_fit)
        if (abs(log(th / th_fit)) < 1e-6) break
      }
      ev <- fn_eval(a + drop(xc %*% b), y, family, th_fit, cx)
      fsat <- fn_fsat(y, family, th_fit)
      n0 <- fn_null(y, family, intercept, th_fit, cx, NULL)
      bfit[, l] <- b; afit[l] <- a; iter[l] <- it_tot; thetas[l] <- th_fit
      devs[l] <- 2 * n * (ev$loss - fsat); nulldevs[l] <- n0$nulldev
      nfit <- l
      if (early.stop && l > 1) {
        dr <- 1 - devs[l] / nulldevs[l]; dr0 <- 1 - devs[l - 1] / nulldevs[l - 1]
        if (dr > 0.999 || (abs(dr - dr0) < 1e-5 && l >= 5)) break
      }
    }
    keep <- seq_len(nfit)
    bfit <- bfit[, keep, drop = FALSE]; afit <- afit[keep]; iter <- iter[keep]
    thetas <- thetas[keep]; devs <- devs[keep]; nulldevs <- nulldevs[keep]
  }

  keep <- seq_len(nfit)
  beta <- bfit / xs
  a0 <- switch(family,
               gaussian = if (intercept) ym - drop(crossprod(xm, beta)) else rep(0, nfit),
               cox = rep(0, nfit),
               afit - drop(crossprod(xm, beta)))
  beta <- matrix(beta, p, nfit, dimnames = list(vnames, paste0("s", keep - 1L)))
  names(a0) <- colnames(beta)
  if (any(iter >= maxit))
    warning(sprintf("maximum iterations (%d) reached at %d lambda value(s); consider increasing 'maxit'",
                    maxit, sum(iter >= maxit)))

  out <- list(a0 = a0, beta = beta,
              df = colSums(beta != 0),
              nfused = apply(beta, 2, fn_count_blocks, seg = seg),
              lambda = lambda[keep], lambda1 = lambda1[keep], lambda2 = lambda2[keep],
              alpha = alphas[keep], theta = if (family == "negbin") thetas else NULL,
              dev.ratio = 1 - devs / nulldevs, nulldev = nulldevs[1],
              iter = iter, npasses = sum(iter), family = family,
              classnames = py$classnames, groups = groups, seg = seg,
              intercept = intercept, standardize = standardize,
              dim = c(p, nfit), nobs = n, call = this.call)
  class(out) <- c("fusednet", paste0("fusednet_", family))
  out
}

# number of distinct non-zero constant blocks (degrees of freedom of the fused lasso)
fn_count_blocks <- function(b, seg = 0L) {
  p <- length(b)
  if (p == 0) return(0L)
  newblock <- c(TRUE, b[-1] != b[-p])
  newblock[seg + 1L] <- TRUE
  sum(newblock & b != 0)
}

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
lambda_interp <- function(lambda, s) {
  # linear interpolation on the lambda scale (as in glmnet)
  if (length(lambda) == 1) {
    nums <- length(s)
    return(list(left = rep(1, nums), right = rep(1, nums), frac = rep(1, nums)))
  }
  k <- length(lambda)
  sfrac <- (lambda[1] - s) / (lambda[1] - lambda[k])
  lambda <- (lambda[1] - lambda) / (lambda[1] - lambda[k])
  sfrac[sfrac < min(lambda)] <- min(lambda)
  sfrac[sfrac > max(lambda)] <- max(lambda)
  coord <- stats::approx(lambda, seq(lambda), sfrac)$y
  left <- floor(coord); right <- ceiling(coord)
  sfrac <- (sfrac - lambda[right]) / (lambda[left] - lambda[right])
  sfrac[left == right] <- 1
  sfrac[abs(lambda[left] - lambda[right]) < .Machine$double.eps] <- 1
  list(left = left, right = right, frac = sfrac)
}

fn_coef_at <- function(object, s) {
  nbeta <- rbind(a0 = object$a0, object$beta)
  rownames(nbeta)[1] <- "(Intercept)"
  if (is.null(s)) return(nbeta)
  lamlist <- lambda_interp(object$lambda, s)
  out <- nbeta[, lamlist$left, drop = FALSE] %*% diag(lamlist$frac, length(s)) +
    nbeta[, lamlist$right, drop = FALSE] %*% diag(1 - lamlist$frac, length(s))
  colnames(out) <- paste0("s", seq_along(s))
  out
}

#' @export
predict.fusednet <- function(object, newx, s = NULL,
                             type = c("link", "response", "class", "coefficients", "nonzero"),
                             exact = FALSE, ...) {
  type <- match.arg(type)
  if (exact && !is.null(s)) {
    which <- match(s, object$lambda, 0)
    if (any(which == 0)) {
      args <- as.list(object$call)[-1]
      env <- parent.frame()
      extra <- list(...)
      args <- lapply(args, eval, envir = env)
      args[names(extra)] <- extra
      args$lambda <- sort(unique(c(object$lambda, s)), decreasing = TRUE)
      args$lambda1 <- NULL; args$lambda2 <- NULL
      object <- do.call(fusednet, args)
    }
  }
  nbeta <- fn_coef_at(object, s)
  if (type == "coefficients") return(nbeta)
  if (type == "nonzero") {
    return(lapply(seq_len(ncol(nbeta)), function(j) which(nbeta[-1, j] != 0)))
  }
  if (missing(newx)) stop("You need to supply a value for 'newx'")
  newx <- as.matrix(newx)
  nfit <- cbind(1, newx) %*% nbeta
  if (object$family == "gaussian" || type == "link") return(nfit)
  if (type == "response") {
    return(if (object$family == "binomial") stats::plogis(nfit) else exp(nfit))
  }
  if (object$family != "binomial") stop("type = \"class\" is only available for family = \"binomial\"")
  cl <- object$classnames
  out <- ifelse(nfit > 0, cl[2], cl[1])
  out
}

#' @export
coef.fusednet <- function(object, s = NULL, exact = FALSE, ...) {
  predict(object, s = s, type = "coefficients", exact = exact, ...)
}

#' @export
print.fusednet <- function(x, digits = max(3, getOption("digits") - 3), ...) {
  cat("\nCall: ", deparse(x$call), "\n\n")
  out <- data.frame(Df = x$df, Blocks = x$nfused,
                    `%Dev` = round(100 * x$dev.ratio, 2),
                    Lambda = signif(x$lambda, digits),
                    Lambda1 = signif(x$lambda1, digits),
                    Lambda2 = signif(x$lambda2, digits),
                    check.names = FALSE)
  if (!is.null(x$theta)) out$Theta <- signif(x$theta, digits)
  print(out, ...)
  invisible(out)
}

#' @export
plot.fusednet <- function(x, xvar = c("lambda", "norm", "dev"),
                          type = c("path", "profile"), s = NULL,
                          label = FALSE, ...) {
  xvar <- match.arg(xvar)
  type <- match.arg(type)
  beta <- x$beta
  if (type == "profile") {
    if (is.null(s)) s <- x$lambda[max(1, round(length(x$lambda) / 2))]
    b <- fn_coef_at(x, s)[-1, , drop = FALSE]
    p <- nrow(b)
    graphics::matplot(seq_len(p), b, type = "s", lty = 1, lwd = 2,
                      xlab = "Coefficient index", ylab = "Coefficient",
                      main = sprintf("Coefficient profile (lambda = %s)",
                                     paste(signif(s, 3), collapse = ", ")), ...)
    graphics::abline(h = 0, col = "grey60", lty = 3)
    return(invisible(b))
  }
  index <- switch(xvar,
                  lambda = log(x$lambda),
                  norm = colSums(abs(beta)),
                  dev = x$dev.ratio)
  xlab <- switch(xvar, lambda = "Log Lambda", norm = "L1 Norm",
                 dev = "Fraction Deviance Explained")
  graphics::matplot(index, t(beta), type = "l", lty = 1, xlab = xlab,
                    ylab = "Coefficients", ...)
  graphics::axis(3, at = index, labels = x$nfused, tick = FALSE, line = 0,
                 cex.axis = 0.7)
  if (label) {
    nr <- nrow(beta)
    pos <- if (xvar == "lambda") 2 else 4
    xpos <- if (xvar == "lambda") min(index) else max(index)
    graphics::text(rep(xpos, nr), beta[, ncol(beta)], rownames(beta),
                   cex = 0.6, pos = pos)
  }
  invisible()
}
