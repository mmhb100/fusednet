## Compare fusednet with CVXR on your own machine.
## install.packages("CVXR")  # if needed
library(fusednet)
library(CVXR)

set.seed(1)
n <- 200; p <- 100
x <- matrix(rnorm(n * p), n)
beta_true <- c(rep(0, 20), rep(2, 20), rep(0, 20), rep(-1, 20), rep(0, 20))
y <- drop(x %*% beta_true + rnorm(n))

lambda1 <- 0.05   # sparsity penalty
lambda2 <- 0.20   # fusion penalty

## ---- CVXR (your current approach) ----
b  <- Variable(p)
b0 <- Variable(1)
obj <- sum_squares(y - b0 - x %*% b) / (2 * n) +
       lambda1 * p_norm(b, 1) +
       lambda2 * p_norm(diff(b), 1)
t_cvxr <- system.time(res <- solve(Problem(Minimize(obj))))[3]
beta_cvxr <- drop(res$getValue(b))

## ---- fusednet (same problem) ----
t_fn <- system.time(fit <- fusednet(x, y, lambda1 = lambda1, lambda2 = lambda2))[3]
beta_fn <- fit$beta[, 1]

cat(sprintf("CVXR: %.3f s   fusednet: %.4f s   speed-up: %.0fx\n", t_cvxr, t_fn, t_cvxr / t_fn))
cat(sprintf("max |beta_CVXR - beta_fusednet| = %.2e\n", max(abs(beta_cvxr - beta_fn))))

## objective values (lower is better)
f <- function(a0, bb) sum((y - a0 - x %*% bb)^2) / (2 * n) + lambda1 * sum(abs(bb)) + lambda2 * sum(abs(diff(bb)))
cat(sprintf("objective CVXR = %.10f   fusednet = %.10f\n",
            f(res$getValue(b0), beta_cvxr), f(fit$a0, beta_fn)))

## ---- a whole path + cross-validation in fusednet (would take very long in CVXR) ----
t_cv <- system.time(cv <- cv.fusednet(x, y, alpha = 0.2, nfolds = 10))[3]
cat(sprintf("10-fold CV over %d lambdas: %.2f s\n", length(cv$lambda), t_cv))
plot(cv)
plot(cv$fusednet.fit, type = "profile", s = cv$lambda.min)
lines(beta_true, col = "grey40", lty = 2)
