#' Cross-validation for fusednet
#' @export
cv.fusednet <- function(x, y, ..., nfolds = 10, foldid = NULL,
                        type.measure = c("default", "mse", "mae", "deviance", "class", "auc", "C"),
                        ncores = 1L, keep = FALSE) {
  type.measure <- match.arg(type.measure)
  this.call <- match.call()
  x <- as.matrix(x)
  n <- nrow(x)
  full <- fusednet(x, y, ...)
  fam <- full$family
  if (type.measure == "default") type.measure <- if (fam == "gaussian") "mse" else "deviance"
  allowed <- switch(fam,
                    gaussian = c("mse", "mae", "deviance"),
                    binomial = c("deviance", "mse", "mae", "class", "auc"),
                    poisson = , negbin = c("deviance", "mse", "mae"),
                    cox = c("deviance", "C"))
  if (!type.measure %in% allowed)
    stop("type.measure '", type.measure, "' is not available for family '", fam,
         "'; use one of: ", paste(allowed, collapse = ", "))
  if (is.null(foldid)) {
    foldid <- sample(rep(seq_len(nfolds), length.out = n))
  } else nfolds <- max(foldid)
  if (nfolds < 3) stop("nfolds must be >= 3")

  py <- fn_prep_y(y, fam, n)
  yy <- py$y
  dots <- list(...)
  dots$lambda <- NULL; dots$nlambda <- NULL; dots$lambda.min.ratio <- NULL
  dots$early.stop <- NULL
  pairs <- !is.null(dots$lambda1) || !is.null(dots$lambda2)
  dots$lambda1 <- NULL; dots$lambda2 <- NULL
  subset_y <- function(idx) if (fam == "cox") y[idx, , drop = FALSE] else y[idx]

  run_fold <- function(k) {
    tr <- foldid != k
    args <- c(list(x = x[tr, , drop = FALSE], y = subset_y(tr)), dots)
    if (pairs) { args$lambda1 <- full$lambda1; args$lambda2 <- full$lambda2 }
    else { args$lambda <- full$lambda }
    args$early.stop <- FALSE
    suppressWarnings(do.call(fusednet, args))
  }
  fits <- if (ncores > 1L && .Platform$OS.type != "windows")
    parallel::mclapply(seq_len(nfolds), run_fold, mc.cores = ncores)
  else lapply(seq_len(nfolds), run_fold)

  nlam <- length(full$lambda)
  w <- as.numeric(table(factor(foldid, levels = seq_len(nfolds))))
  fitmat <- NULL

  if (fam == "cox") {
    # per-fold measures on the held-out data
    foldm <- matrix(NA_real_, nlam, nfolds)
    for (k in seq_len(nfolds)) {
      fk <- fits[[k]]; te <- foldid == k; tr <- !te
      eta_all <- x %*% fk$beta[, seq_len(nlam), drop = FALSE]
      if (type.measure == "deviance") {
        # Verweij & van Houwelingen (1993): l(all) - l(training), per held-out event
        cx_all <- fn_cox_struct(py$time); cx_tr <- fn_cox_struct(py$time[tr])
        dk <- max(sum(yy[te]), 1)
        for (l in seq_len(nlam)) {
          ll_all <- -n * fn_eval(eta_all[, l], yy, "cox", 1, cx_all)$loss
          ll_tr <- -sum(tr) * fn_eval(eta_all[tr, l], yy[tr], "cox", 1, cx_tr)$loss
          foldm[l, k] <- -2 * (ll_all - ll_tr) / dk
        }
      } else {
        for (l in seq_len(nlam))
          foldm[l, k] <- fn_cindex(eta_all[te, l], py$time[te], yy[te])
      }
    }
    if (type.measure == "deviance") w <- vapply(seq_len(nfolds), function(k) max(sum(yy[foldid == k]), 1), 0)
    cvm <- apply(foldm, 1, stats::weighted.mean, w = w, na.rm = TRUE)
    cvsd <- sqrt(apply((foldm - cvm)^2, 1, stats::weighted.mean, w = w, na.rm = TRUE) / (nfolds - 1))
  } else {
    fitmat <- matrix(NA_real_, n, nlam)
    thmat <- matrix(1, n, nlam)
    for (k in seq_len(nfolds)) {
      fk <- fits[[k]]; te <- foldid == k
      fitmat[te, ] <- predict(fk, x[te, , drop = FALSE], type = "link")[, seq_len(nlam)]
      if (fam == "negbin") thmat[te, ] <- rep(fk$theta[seq_len(nlam)], each = sum(te))
    }
    if (type.measure == "auc") {
      foldm <- sapply(seq_len(nfolds), function(k) {
        idx <- foldid == k
        apply(fitmat[idx, , drop = FALSE], 2, fn_auc, y = yy[idx])
      })
      foldm <- matrix(foldm, nlam)
      cvm <- apply(foldm, 1, stats::weighted.mean, w = w, na.rm = TRUE)
    } else {
      Lmat <- fn_cv_loss(fitmat, yy, fam, type.measure, thmat)
      foldm <- sapply(seq_len(nfolds), function(k) colMeans(Lmat[foldid == k, , drop = FALSE]))
      foldm <- matrix(foldm, nlam)
      cvm <- colMeans(Lmat)
    }
    cvsd <- sqrt(apply((foldm - cvm)^2, 1, stats::weighted.mean, w = w, na.rm = TRUE) / (nfolds - 1))
  }

  maximize <- type.measure %in% c("auc", "C")
  idmin <- if (maximize) which.max(cvm) else which.min(cvm)
  lambda.min <- full$lambda[idmin]
  thr <- if (maximize) cvm[idmin] - cvsd[idmin] else cvm[idmin] + cvsd[idmin]
  ok <- if (maximize) cvm >= thr else cvm <= thr
  ok[is.na(ok)] <- FALSE
  lambda.1se <- max(full$lambda[ok])
  name <- switch(type.measure, mse = "Mean-Squared Error", mae = "Mean Absolute Error",
                 deviance = switch(fam, gaussian = "Mean-Squared Error",
                                   binomial = "Binomial Deviance", poisson = "Poisson Deviance",
                                   negbin = "Negative Binomial -2 Log-Likelihood",
                                   cox = "Partial Likelihood Deviance"),
                 class = "Misclassification Error", auc = "AUC", C = "C-index")
  out <- list(lambda = full$lambda, cvm = cvm, cvsd = cvsd,
              cvup = cvm + cvsd, cvlo = cvm - cvsd,
              nzero = full$df, nfused = full$nfused, name = name,
              fusednet.fit = full, lambda.min = lambda.min, lambda.1se = lambda.1se,
              index = c(min = idmin, `1se` = which(full$lambda == lambda.1se)[1]),
              foldid = foldid, call = this.call)
  if (keep) out$fit.preval <- fitmat
  class(out) <- "cv.fusednet"
  out
}

# observation-level CV losses on the held-out linear predictor eta
fn_cv_loss <- function(eta, y, fam, type, theta) {
  mu <- switch(fam, gaussian = eta, binomial = stats::plogis(eta), exp(eta))
  if (type == "mse") return((y - mu)^2)
  if (type == "mae") return(abs(y - mu))
  if (fam == "gaussian") return((y - mu)^2)
  if (fam == "binomial") {
    pr <- pmin(pmax(mu, 1e-5), 1 - 1e-5)
    if (type == "class") return(((pr > 0.5) != (y == 1)) * 1)
    return(-2 * (y * log(pr) + (1 - y) * log(1 - pr)))
  }
  y <- matrix(y, nrow(mu), ncol(mu))             # ifelse() keeps the shape of its test
  ylogy <- ifelse(y > 0, y * log(pmax(y, 1e-300) / mu), 0)
  if (fam == "poisson") return(2 * (ylogy - (y - mu)))
  # negative binomial: -2 * full log-likelihood. Unlike the deviance it is
  # comparable across different theta (theta is re-estimated at each lambda);
  # for a fixed theta it differs from the deviance only by a constant.
  -2 * (lgamma(y + theta) - lgamma(theta) - lgamma(y + 1) +
          theta * log(theta / (theta + mu)) + y * log(mu / (theta + mu)))
}

# Harrell's concordance index (higher risk score = shorter survival)
fn_cindex <- function(eta, time, status) {
  ev <- which(status == 1)
  if (length(ev) == 0) return(NA_real_)
  conc <- 0; comp <- 0
  for (i in ev) {
    j <- which(time > time[i])
    if (!length(j)) next
    comp <- comp + length(j)
    conc <- conc + sum(eta[i] > eta[j]) + 0.5 * sum(eta[i] == eta[j])
  }
  if (comp == 0) NA_real_ else conc / comp
}

fn_auc <- function(eta, y) {
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(eta)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

fn_resolve_s <- function(object, s) {
  if (is.character(s)) {
    s <- match.arg(s, c("lambda.1se", "lambda.min"))
    return(object[[s]])
  }
  s
}

#' @export
coef.cv.fusednet <- function(object, s = c("lambda.1se", "lambda.min"), ...) {
  if (missing(s)) s <- "lambda.1se"
  coef(object$fusednet.fit, s = fn_resolve_s(object, s), ...)
}

#' @export
predict.cv.fusednet <- function(object, newx, s = c("lambda.1se", "lambda.min"), ...) {
  if (missing(s)) s <- "lambda.1se"
  predict(object$fusednet.fit, newx, s = fn_resolve_s(object, s), ...)
}

#' @export
print.cv.fusednet <- function(x, digits = max(3, getOption("digits") - 3), ...) {
  cat("\nCall: ", deparse(x$call), "\n\n")
  cat("Measure:", x$name, "\n\n")
  idx <- x$index
  out <- data.frame(Lambda = signif(x$lambda[idx], digits),
                    Index = idx,
                    Measure = signif(x$cvm[idx], digits),
                    SE = signif(x$cvsd[idx], digits),
                    Nonzero = x$nzero[idx], Blocks = x$nfused[idx],
                    row.names = c("min", "1se"))
  print(out, ...)
  invisible(out)
}

#' @export
plot.cv.fusednet <- function(x, sign.lambda = 1, ...) {
  xl <- sign.lambda * log(x$lambda)
  graphics::plot(xl, x$cvm, ylim = range(x$cvup, x$cvlo, na.rm = TRUE),
                 xlab = if (sign.lambda > 0) "Log(Lambda)" else "-Log(Lambda)",
                 ylab = x$name, type = "n", ...)
  graphics::segments(xl, x$cvlo, xl, x$cvup, col = "darkgrey")
  graphics::points(xl, x$cvm, pch = 20, col = "red")
  graphics::axis(3, at = xl, labels = x$nfused, tick = FALSE, line = 0, cex.axis = 0.7)
  graphics::abline(v = sign.lambda * log(c(x$lambda.min, x$lambda.1se)), lty = 3)
  invisible()
}
