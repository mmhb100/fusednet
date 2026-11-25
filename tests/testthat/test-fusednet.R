# Brute-force reference for the fused lasso signal approximator: dual projected gradient
ref_tv <- function(y, lam, iters = 20000) {
  n <- length(y); u <- numeric(n - 1)
  for (k in seq_len(iters)) {
    x <- y - c(-u, 0) - c(0, u)          # x = y - D'u
    u <- pmin(pmax(u + 0.25 * diff(x), -lam), lam)
  }
  y - c(-u, 0) - c(0, u)
}

test_that("exact prox (Condat TV + soft threshold) matches a reference", {
  set.seed(1)
  for (i in 1:20) {
    p <- sample(2:40, 1); y <- cumsum(rnorm(p))
    l1 <- runif(1); l2 <- runif(1, 0, 2)
    ref <- ref_tv(y, l2)
    ref <- sign(ref) * pmax(abs(ref) - l1, 0)
    expect_equal(flsa(y, l1, l2), ref, tolerance = 1e-6)
  }
  expect_equal(flsa(c(1, 2, 3), 0, 0), c(1, 2, 3))
  expect_equal(flsa(5, 1, 3), 4)
})

test_that("gaussian path satisfies KKT conditions (n > p and p > n)", {
  set.seed(2)
  for (cfg in list(c(100, 20), c(30, 60))) {
    n <- cfg[1]; p <- cfg[2]
    X <- matrix(rnorm(n * p), n); b <- c(rep(0, 5), rep(2, 5), rep(0, p - 10))
    y <- drop(X %*% b + rnorm(n))
    fit <- fusednet(X, y, alpha = 0.4, nlambda = 30)
    expect_lt(max(kkt.fusednet(fit, X, y)), 1e-5)
    fitn <- fusednet(X, y, alpha = 0.4, nlambda = 30, type.solver = "naive")
    expect_equal(fit$beta, fitn$beta, tolerance = 1e-5)
  }
})

test_that("first lambda gives the null model and the path is monotone", {
  set.seed(3)
  X <- matrix(rnorm(400), 40); y <- rnorm(40)
  fit <- fusednet(X, y, alpha = 0.5)
  expect_true(all(fit$beta[, 1] == 0))
  expect_true(any(fit$beta[, 2] != 0))
  expect_true(all(diff(fit$lambda) < 0))
})

test_that("alpha = 1 reproduces glmnet's lasso", {
  skip_if_not_installed("glmnet")
  set.seed(4)
  X <- matrix(rnorm(80 * 15), 80); y <- drop(X[, 1:3] %*% c(1, -1, 2) + rnorm(80))
  fit <- fusednet(X, y, alpha = 1, nlambda = 20)
  g <- glmnet::glmnet(X, y, lambda = fit$lambda, standardize = FALSE, thresh = 1e-16)
  expect_equal(unname(as.matrix(g$beta)), unname(fit$beta), tolerance = 1e-5)
  expect_equal(unname(g$a0), unname(fit$a0), tolerance = 1e-5)
  yb <- rbinom(80, 1, plogis(X[, 1]))
  fb <- fusednet(X, yb, family = "binomial", alpha = 1, nlambda = 20)
  gb <- glmnet::glmnet(X, yb, family = "binomial", lambda = fb$lambda, standardize = FALSE, thresh = 1e-16)
  expect_equal(unname(as.matrix(gb$beta)), unname(fb$beta), tolerance = 1e-4)
})

test_that("binomial, standardize, groups, no-intercept and lambda pairs are optimal", {
  set.seed(5)
  n <- 120; p <- 30; X <- matrix(rnorm(n * p), n) %*% diag(runif(p, 0.5, 3))
  yb <- factor(ifelse(rbinom(n, 1, plogis(rowSums(X[, 5:10]) / 3)) == 1, "yes", "no"))
  fb <- fusednet(X, yb, family = "binomial", alpha = 0.3, nlambda = 25)
  expect_lt(max(kkt.fusednet(fb, X, yb)), 1e-5)
  expect_true(all(predict(fb, X[1:3, ], type = "class") %in% c("no", "yes")))
  y <- drop(X %*% c(rep(0, 10), rep(1, 10), rep(0, 10)) + rnorm(n))
  fs <- fusednet(X, y, alpha = 0.6, standardize = TRUE, nlambda = 25)
  expect_lt(max(kkt.fusednet(fs, X, y)), 1e-5)
  fg <- fusednet(X, y, alpha = 0, groups = rep(1:3, each = 10), nlambda = 25)
  expect_lt(max(kkt.fusednet(fg, X, y)), 1e-5)
  fi <- fusednet(X, y, alpha = 0.5, intercept = FALSE, nlambda = 25)
  expect_true(all(fi$a0 == 0))
  expect_lt(max(kkt.fusednet(fi, X, y)), 1e-5)
  fp <- fusednet(X, y, lambda1 = c(0.2, 0.1, 0.01), lambda2 = c(0.5, 0.2, 0.05))
  expect_equal(fp$lambda1, c(0.2, 0.1, 0.01))
  expect_lt(max(kkt.fusednet(fp, X, y)), 1e-5)
})

test_that("predict/coef/cv work", {
  set.seed(6)
  X <- matrix(rnorm(100 * 20), 100); y <- drop(X[, 5:9] %*% rep(1, 5) + rnorm(100))
  fit <- fusednet(X, y)
  pr <- predict(fit, X[1:5, ], s = fit$lambda[c(3, 10)])
  expect_equal(dim(pr), c(5, 2))
  expect_equal(drop(pr[, 2]), drop(fit$a0[10] + X[1:5, ] %*% fit$beta[, 10]))
  cf <- coef(fit, s = mean(fit$lambda[4:5]))
  expect_equal(nrow(cf), 21)
  cv <- cv.fusednet(X, y, nfolds = 5)
  expect_true(cv$lambda.1se >= cv$lambda.min)
  expect_equal(length(cv$cvm), length(cv$lambda))
  expect_equal(nrow(coef(cv, s = "lambda.min")), 21)
  yb <- rbinom(100, 1, plogis(X[, 1]))
  cvb <- cv.fusednet(X, yb, family = "binomial", type.measure = "auc", nfolds = 5)
  expect_true(all(cvb$cvm >= 0 & cvb$cvm <= 1, na.rm = TRUE))
})

# ---------------------------------------------------------------------------
# Poisson, negative binomial and Cox families
# ---------------------------------------------------------------------------
sim_glm <- function(n = 200, p = 30, seed = 7) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n)
  b <- rep(0, p); b[8:15] <- 0.3; b[22:25] <- -0.4
  list(X = X, eta = drop(X %*% b))
}

test_that("poisson: KKT optimal and alpha = 1 matches glmnet", {
  d <- sim_glm(); y <- rpois(nrow(d$X), exp(d$eta))
  fit <- fusednet(d$X, y, family = "poisson", alpha = 0.5, nlambda = 30)
  expect_lt(max(kkt.fusednet(fit, d$X, y)), 1e-5)
  expect_true(all(predict(fit, d$X[1:3, ], s = fit$lambda[5], type = "response") > 0))
  skip_if_not_installed("glmnet")
  f1 <- fusednet(d$X, y, family = "poisson", alpha = 1, nlambda = 15)
  g1 <- glmnet::glmnet(d$X, y, family = "poisson", lambda = f1$lambda,
                       standardize = FALSE, thresh = 1e-16, maxit = 1e7)
  expect_equal(unname(as.matrix(g1$beta)), unname(f1$beta), tolerance = 1e-5)
  expect_equal(unname(g1$dev.ratio), unname(f1$dev.ratio), tolerance = 1e-6)
})

test_that("negative binomial: theta estimation matches MASS and fits are optimal", {
  d <- sim_glm(); y <- rnbinom(nrow(d$X), size = 2, mu = exp(1 + d$eta))
  f0 <- fusednet(d$X[, 1:4], y, family = "negbin", lambda1 = 0, lambda2 = 0, thresh = 1e-10)
  skip_if_not_installed("MASS")
  g0 <- MASS::glm.nb(y ~ d$X[, 1:4])
  expect_equal(unname(c(f0$a0, f0$beta[, 1])), unname(coef(g0)), tolerance = 1e-6)
  expect_equal(f0$theta, g0$theta, tolerance = 1e-6)
  expect_equal(nb.theta(y, fitted(g0)), MASS::theta.ml(y, fitted(g0))[1], tolerance = 1e-6,
               ignore_attr = TRUE)
  fit <- fusednet(d$X, y, family = "negbin", alpha = 0.5, nlambda = 20)
  expect_lt(max(kkt.fusednet(fit, d$X, y)), 1e-5)
  ff <- fusednet(d$X, y, family = "negbin", theta = 2, alpha = 0.3, nlambda = 20)
  expect_true(all(ff$theta == 2))
  expect_lt(max(kkt.fusednet(ff, d$X, y)), 1e-5)
})

test_that("cox: matches coxph (Breslow) and glmnet, with ties", {
  skip_if_not_installed("survival")
  d <- sim_glm()
  n <- nrow(d$X)
  tt <- round(rexp(n, exp(d$eta)), 1) + 0.1; cc <- rexp(n, 0.3)
  y <- survival::Surv(pmin(tt, cc), as.numeric(tt <= cc))
  f0 <- fusednet(d$X[, 1:4], y, family = "cox", lambda1 = 0, lambda2 = 0, thresh = 1e-10)
  cp <- survival::coxph(y ~ d$X[, 1:4], ties = "breslow")
  expect_equal(unname(f0$beta[, 1]), unname(coef(cp)), tolerance = 1e-7)
  fit <- fusednet(d$X, y, family = "cox", alpha = 0.5, nlambda = 25)
  expect_true(all(fit$a0 == 0))
  expect_lt(max(kkt.fusednet(fit, d$X, y)), 1e-5)
  skip_if_not_installed("glmnet")
  f1 <- fusednet(d$X, y, family = "cox", alpha = 1, nlambda = 15)
  g1 <- glmnet::glmnet(d$X, y, family = "cox", lambda = f1$lambda,
                       standardize = FALSE, thresh = 1e-16, maxit = 1e7)
  expect_equal(unname(as.matrix(g1$beta)), unname(f1$beta), tolerance = 1e-5)
})

test_that("cv works for poisson, negbin and cox", {
  skip_if_not_installed("survival")
  d <- sim_glm(n = 150)
  n <- nrow(d$X); fid <- rep(1:4, length.out = n)
  cvp <- cv.fusednet(d$X, rpois(n, exp(d$eta)), family = "poisson", foldid = fid)
  expect_true(cvp$lambda.1se >= cvp$lambda.min)
  cvn <- cv.fusednet(d$X, rnbinom(n, size = 2, mu = exp(d$eta)), family = "negbin",
                     theta = 2, foldid = fid)
  expect_equal(length(cvn$cvm), length(cvn$lambda))
  tt <- rexp(n, exp(d$eta)); cc <- rexp(n, 0.3)
  y <- survival::Surv(pmin(tt, cc), as.numeric(tt <= cc))
  cvc <- cv.fusednet(d$X, y, family = "cox", foldid = fid)
  expect_true(all(is.finite(cvc$cvm)))
  cvC <- cv.fusednet(d$X, y, family = "cox", foldid = fid, type.measure = "C")
  expect_true(all(cvC$cvm >= 0 & cvC$cvm <= 1, na.rm = TRUE))
  expect_error(cv.fusednet(d$X, y, family = "cox", type.measure = "mse"))
})

test_that("relax.fusednet refits block values without penalty", {
  set.seed(11)
  X <- matrix(rnorm(150 * 30), 150)
  b <- c(rep(0, 10), rep(1, 10), rep(0, 10))
  y <- drop(X %*% b + rnorm(150))
  fit <- fusednet(X, y, alpha = 0.5, nlambda = 40)
  s <- fit$lambda[15]
  bs <- coef(fit, s = s)[-1, 1]
  r <- relax.fusednet(fit, X, y, s = s)
  # gaussian: the relaxed fit equals lm() on the block sums
  id <- cumsum(c(TRUE, diff(bs) != 0)); blocks <- unique(id[bs != 0])
  Z <- sapply(blocks, function(g) rowSums(X[, id == g, drop = FALSE]))
  co <- coef(lm(y ~ Z))
  expect_equal(unname(r[1, 1]), unname(co[1]), tolerance = 1e-6)
  expect_equal(unname(r[-1, 1][bs != 0]), unname(co[-1][match(id[bs != 0], blocks)]), tolerance = 1e-6)
  expect_true(all(r[-1, 1][bs == 0] == 0))
  expect_equal(relax.fusednet(fit, X, y, s = s, gamma = 1), coef(fit, s = s))
})
