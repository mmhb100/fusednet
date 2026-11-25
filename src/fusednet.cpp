// fusednet: fast sparse fused lasso for linear and logistic regression
//
// Objective (for one lambda pair):
//   minimize_{a0, beta}  F(a0, beta) + lambda1 * sum_j |beta_j|
//                                    + lambda2 * sum_{j=1}^{p-1} |beta_{j+1} - beta_j|
//   gaussian : F = 1/(2n) * sum_i (y_i - a0 - x_i' beta)^2
//   binomial : F = 1/n * sum_i [ log(1 + exp(eta_i)) - y_i * eta_i ],  eta = a0 + X beta
//
// Solver: accelerated proximal gradient (FISTA) with backtracking and adaptive
// (gradient-based) restart. The proximal operator of the sparse fused lasso
// penalty is computed exactly in O(p):
//   prox(v) = soft_threshold( TV1D_denoise(v, lambda2/L), lambda1/L )
// (Friedman, Hastie, Hoefling & Tibshirani 2007, Prop. 1), using Condat's
// direct 1D total-variation algorithm (Condat 2013).
//
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// ---------------------------------------------------------------------------
// Condat (2013) "A direct algorithm for 1D total variation denoising".
// Solves  min_x 0.5 * ||x - input||^2 + lambda * sum |x_{k+1} - x_k|
// ---------------------------------------------------------------------------
static void tv1d_denoise(const double* input, double* output, const int width,
                         const double lambda) {
  if (width <= 0) return;
  if (lambda <= 0.0) {
    for (int i = 0; i < width; ++i) output[i] = input[i];
    return;
  }
  int k = 0, k0 = 0;
  double umin = lambda, umax = -lambda;
  double vmin = input[0] - lambda, vmax = input[0] + lambda;
  int kplus = 0, kminus = 0;
  const double twolambda = 2.0 * lambda;
  const double minlambda = -lambda;
  for (;;) {
    while (k == width - 1) {
      if (umin < 0.0) {
        do output[k0++] = vmin; while (k0 <= kminus);
        umax = (vmin = input[kminus = k = k0]) + (umin = lambda) - vmax;
      } else if (umax > 0.0) {
        do output[k0++] = vmax; while (k0 <= kplus);
        umin = (vmax = input[kplus = k = k0]) + (umax = minlambda) - vmin;
      } else {
        vmin += umin / (k - k0 + 1);
        do output[k0++] = vmin; while (k0 <= k);
        return;
      }
    }
    if ((umin += input[k + 1] - vmin) < minlambda) {
      do output[k0++] = vmin; while (k0 <= kminus);
      vmax = (vmin = input[kplus = kminus = k = k0]) + twolambda;
      umin = lambda;
      umax = minlambda;
    } else if ((umax += input[k + 1] - vmax) > lambda) {
      do output[k0++] = vmax; while (k0 <= kplus);
      vmin = (vmax = input[kplus = kminus = k = k0]) - twolambda;
      umin = lambda;
      umax = minlambda;
    } else {
      k++;
      if (umin >= lambda) {
        vmin += (umin - lambda) / ((kminus = k) - k0 + 1);
        umin = lambda;
      }
      if (umax <= minlambda) {
        vmax += (umax + lambda) / ((kplus = k) - k0 + 1);
        umax = minlambda;
      }
    }
  }
}

// Exact prox of  l1*||b||_1 + l2*TV(b)  applied within each fusion segment.
// seg_start: 0-based start index of each segment (sorted, first = 0).
static void prox_sfl(const arma::vec& v, arma::vec& out, double l1, double l2,
                     const std::vector<int>& seg_start) {
  const int p = v.n_elem;
  if (l2 > 0.0) {
    const int ns = seg_start.size();
    for (int s = 0; s < ns; ++s) {
      const int a = seg_start[s];
      const int b = (s + 1 < ns) ? seg_start[s + 1] : p;
      tv1d_denoise(v.memptr() + a, out.memptr() + a, b - a, l2);
    }
  } else {
    out = v;
  }
  if (l1 > 0.0) {
    double* o = out.memptr();
    for (int j = 0; j < p; ++j) {
      const double z = o[j];
      o[j] = (z > l1) ? z - l1 : ((z < -l1) ? z + l1 : 0.0);
    }
  }
}

static inline double softplus(double x) {
  return (x > 0.0) ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}
static inline double sigmoid(double x) {
  if (x >= 0.0) { const double e = std::exp(-x); return 1.0 / (1.0 + e); }
  const double e = std::exp(x); return e / (1.0 + e);
}

static double power_L_cov(const arma::mat& A, int maxit = 100) {
  const int p = A.n_cols;
  arma::vec v(p, arma::fill::ones);
  v /= std::sqrt((double)p);
  double lam = 0.0;
  for (int it = 0; it < maxit; ++it) {
    arma::vec w = A * v;
    double nw = arma::norm(w);
    if (nw <= 0.0) return 1e-12;
    v = w / nw;
    if (it > 5 && std::fabs(nw - lam) <= 1e-4 * nw) { lam = nw; break; }
    lam = nw;
  }
  return lam;
}

// ---------------------------------------------------------------------------
// KKT / feasibility certificate.
// Checks whether there exist s_k in S_k, u_k in U_k, e_k in [-eps, eps] with
//   g_k = l1*s_k + l2*(u_{k-1} - u_k) + e_k     (u_0 = u_p = 0 at segment ends)
// where S_k = {sign(beta_k)} if beta_k != 0 else [-1,1],
//       U_k = {sign(beta_{k+1}-beta_k)} if different, else [-1,1].
// g is the negative gradient of the smooth loss at beta.
// ---------------------------------------------------------------------------
static bool kkt_feasible(const arma::vec& g, const arma::vec& beta, double l1,
                         double l2, double eps, double ztol,
                         const std::vector<int>& seg_start) {
  const int p = g.n_elem;
  const int ns = seg_start.size();
  for (int s = 0; s < ns; ++s) {
    const int a = seg_start[s];
    const int b = (s + 1 < ns) ? seg_start[s + 1] : p;
    double lo = 0.0, hi = 0.0;  // feasible interval for u_{k-1}
    for (int k = a; k < b; ++k) {
      double smin = -1.0, smax = 1.0;
      if (beta[k] > ztol) { smin = smax = 1.0; }
      else if (beta[k] < -ztol) { smin = smax = -1.0; }
      if (l2 <= 0.0) {
        // need |g_k - l1 s| <= eps for some s in [smin, smax]
        const double cmin = g[k] - l1 * smax, cmax = g[k] - l1 * smin;
        if (cmin > eps || cmax < -eps) return false;
        continue;
      }
      // u_k = u_{k-1} - (g_k - l1 s - e) / l2
      double nlo = lo - (g[k] - l1 * smin + eps) / l2;
      double nhi = hi - (g[k] - l1 * smax - eps) / l2;
      if (k == b - 1) {
        // segment end: u must equal 0
        if (nlo > 1e-12 || nhi < -1e-12) return false;
      } else {
        double umin = -1.0, umax = 1.0;
        const double d = beta[k + 1] - beta[k];
        if (d > ztol) { umin = umax = 1.0; }
        else if (d < -ztol) { umin = umax = -1.0; }
        lo = std::max(nlo, umin);
        hi = std::min(nhi, umax);
        if (lo > hi + 1e-12) return false;
      }
    }
  }
  return true;
}

static std::vector<int> make_segments(const IntegerVector& seg) {
  std::vector<int> s(seg.begin(), seg.end());
  if (s.empty() || s[0] != 0) s.insert(s.begin(), 0);
  return s;
}

// smallest eps making the KKT system feasible (bisection)
// [[Rcpp::export(.kkt_violation)]]
double kkt_violation(const arma::vec& g, const arma::vec& beta, double l1,
                     double l2, double ztol, IntegerVector seg) {
  std::vector<int> ss = make_segments(seg);
  if (kkt_feasible(g, beta, l1, l2, 0.0, ztol, ss)) return 0.0;
  double hi = arma::abs(g).max() + l1 + 2.0 * l2 + 1.0;
  double lo = 0.0;
  for (int it = 0; it < 100; ++it) {
    double mid = 0.5 * (lo + hi);
    if (kkt_feasible(g, beta, l1, l2, mid, ztol, ss)) hi = mid; else lo = mid;
    if (hi - lo <= 1e-14 * (1.0 + hi)) break;
  }
  return hi;
}

// Smallest lambda such that beta = 0 is optimal with l1 = alpha*lambda,
// l2 = (1-alpha)*lambda, given negative gradient g at beta = 0 (alpha > 0).
// [[Rcpp::export(.lambda_max)]]
double lambda_max_cpp(const arma::vec& g, double alpha, IntegerVector seg) {
  std::vector<int> ss = make_segments(seg);
  const arma::vec zero(g.n_elem, arma::fill::zeros);
  double hi = arma::abs(g).max() / alpha;
  if (hi <= 0.0) return 0.0;
  double lo = 0.0;
  for (int it = 0; it < 200; ++it) {
    double mid = 0.5 * (lo + hi);
    if (kkt_feasible(g, zero, alpha * mid, (1.0 - alpha) * mid, 0.0, 0.0, ss))
      hi = mid;
    else
      lo = mid;
    if (hi - lo <= 1e-10 * hi) break;
  }
  return hi;
}

// [[Rcpp::export(.tv1d)]]
NumericVector tv1d_R(NumericVector y, double lambda) {
  NumericVector out(y.size());
  tv1d_denoise(y.begin(), out.begin(), y.size(), lambda);
  return out;
}

// [[Rcpp::export(.prox_sfl)]]
arma::vec prox_sfl_R(const arma::vec& v, double l1, double l2, IntegerVector seg) {
  std::vector<int> ss = make_segments(seg);
  arma::vec out(v.n_elem);
  prox_sfl(v, out, l1, l2, ss);
  return out;
}

// ---------------------------------------------------------------------------
// Largest eigenvalue of M' W M / n, M = [1, X] (if icpt) or X, W = diag(w).
// Used as the starting step-size constant; backtracking corrects it.
// ---------------------------------------------------------------------------
static double power_L_weighted(const arma::mat& X, bool icpt, const arma::vec& w,
                               int maxit = 100) {
  const int n = X.n_rows, p = X.n_cols;
  arma::vec v(p, arma::fill::ones);
  double v0 = icpt ? 1.0 : 0.0;
  double nv = std::sqrt(arma::dot(v, v) + v0 * v0);
  v /= nv; v0 /= nv;
  double lam = 0.0;
  for (int it = 0; it < maxit; ++it) {
    arma::vec Mv = X * v;
    if (icpt) Mv += v0;
    Mv %= w;
    arma::vec u = X.t() * Mv / n;
    double u0 = icpt ? arma::accu(Mv) / n : 0.0;
    double nu = std::sqrt(arma::dot(u, u) + u0 * u0);
    if (nu <= 0.0) return 1e-12;
    v = u / nu; v0 = u0 / nu;
    if (it > 5 && std::fabs(nu - lam) <= 1e-4 * nu) { lam = nu; break; }
    lam = nu;
  }
  return lam;
}

// expm1(d) - d, accurate for small d
static inline double em1md(double d) {
  if (std::fabs(d) < 1e-3) return d * d * (0.5 + d * (1.0 / 6.0 + d / 24.0));
  return std::expm1(d) - d;
}
// log1p(t) - t, accurate for small t (t > -1)
static inline double log1pmx(double t) {
  if (std::fabs(t) < 1e-4) return t * t * (-0.5 + t * (1.0 / 3.0 - 0.25 * t));
  return std::log1p(t) - t;
}
// Bregman divergence of softplus(. - c) at a with step dl, accurate for small dl
static inline double sp_breg(double a, double dl, double c) {
  const double sg = sigmoid(a - c);
  if (std::fabs(dl) < 1e-3) {
    const double w = sg * (1.0 - sg);
    return 0.5 * w * dl * dl + w * (1.0 - 2.0 * sg) * dl * dl * dl / 6.0;
  }
  return softplus(a + dl - c) - softplus(a - c) - sg * dl;
}

// ---------------------------------------------------------------------------
// Path solver
// family: 0 gaussian, 1 binomial, 2 poisson, 3 negative binomial (fixed theta),
//         4 cox (Breslow partial likelihood; y = status, ord/gstart/gend = ties)
// For gaussian with intercept the R side centres X and y and passes icpt=false.
// The loss is normalised by n. Returned 'f' is the loss at each solution.
// ---------------------------------------------------------------------------
// [[Rcpp::export(.fusednet_path)]]
List fusednet_path(const arma::mat& X, const arma::vec& y, int family,
                   const arma::vec& lam1, const arma::vec& lam2, bool icpt,
                   double thresh, int maxit, bool covmode, double nulldev,
                   double fsat, IntegerVector seg, bool early_stop,
                   double a0_init, const arma::vec& beta_init,
                   const arma::vec& wL, double theta,
                   IntegerVector ord, IntegerVector gstart, IntegerVector gend) {
  const int n = X.n_rows, p = X.n_cols, nlam = lam1.n_elem;
  const std::vector<int> ss = make_segments(seg);
  enum { GAUS = 0, BINO = 1, POIS = 2, NEGB = 3, COX = 4 };
  const bool gaus = (family == GAUS);
  if (!gaus) covmode = false;
  if (family == COX) icpt = false;
  // step size may also shrink back: Poisson/Cox curvature is unbounded, and the NB
  // bound (y+theta)/4 is very loose when theta is large
  const bool adapt = (family == POIS || family == COX || family == NEGB);
  const double cth = (family == NEGB) ? std::log(theta) : 0.0;

  // ---- Cox bookkeeping (sorted by ascending time) ----
  arma::vec dsort, ws(n), Dn(n), S(n), S1(n), S2(n);
  if (family == COX) {
    dsort.set_size(n);
    for (int k = 0; k < n; ++k) dsort[k] = y[ord[k]];
  }
  // fills ws (weights exp(eta - m) in sorted order) and Dn (risk-set sums); returns m
  auto cox_prep = [&](const arma::vec& eta) -> double {
    const double m = eta.max();
    for (int k = 0; k < n; ++k) ws[k] = std::exp(eta[ord[k]] - m);
    double acc = 0.0;
    for (int k = n - 1; k >= 0; --k) { acc += ws[k]; S[k] = acc; }
    for (int k = 0; k < n; ++k) Dn[k] = S[gstart[k]];
    return m;
  };

  arma::mat A;
  arma::vec bvec;
  double yy = 0.0;
  double L;
  if (covmode) {
    A = X.t() * X / (double)n;
    bvec = X.t() * y / (double)n;
    yy = arma::dot(y, y) / n;
    L = power_L_cov(A);
  } else {
    L = power_L_weighted(X, icpt, wL);
  }
  L = std::max(L, 1e-10);
  const double Lfloor = 1e-6 * L;
  double scale = gaus ? arma::dot(y, y) / n : 0.25;
  if (gaus && icpt) {
    const double m = arma::mean(y);
    scale = arma::dot(y - m, y - m) / n;
  }
  if (scale <= 0.0) scale = 1.0;

  arma::mat betas(p, nlam, arma::fill::zeros);
  arma::vec a0s(nlam, arma::fill::zeros);
  IntegerVector iters(nlam);
  NumericVector fs(nlam), devs(nlam);

  arma::vec beta(p, arma::fill::zeros), betap(p, arma::fill::zeros);
  if ((int)beta_init.n_elem == p) beta = beta_init;
  double a0 = icpt ? a0_init : 0.0, a0p = a0;

  auto compute_q = [&](const arma::vec& b, double c) -> arma::vec {
    if (covmode) return A * b;
    arma::uvec nz = arma::find(b);
    arma::vec e;
    if (nz.n_elem == 0) e.zeros(n);
    else if (nz.n_elem < 0.5 * p) e = X.cols(nz) * b.elem(nz);
    else e = X * b;
    if (icpt) e += c;
    return e;
  };
  auto loss = [&](const arma::vec& q, const arma::vec& b) -> double {
    if (covmode) return 0.5 * arma::dot(b, q) - arma::dot(bvec, b) + 0.5 * yy;
    double s = 0.0;
    switch (family) {
    case GAUS: for (int i = 0; i < n; ++i) { const double r = y[i] - q[i]; s += 0.5 * r * r; } break;
    case BINO: for (int i = 0; i < n; ++i) s += softplus(q[i]) - y[i] * q[i]; break;
    case POIS: for (int i = 0; i < n; ++i) s += std::exp(q[i]) - y[i] * q[i]; break;
    case NEGB: for (int i = 0; i < n; ++i)
        s += (y[i] + theta) * (cth + softplus(q[i] - cth)) - y[i] * q[i];
      break;
    case COX: {
      const double m = cox_prep(q);
      for (int k = 0; k < n; ++k)
        if (dsort[k] > 0) s -= dsort[k] * (q[ord[k]] - m - std::log(Dn[k]));
      break; }
    }
    return s / n;
  };
  // derivative of n*loss w.r.t. eta -> r; gradient g = X'r/n, g0 = mean(r)
  auto gradient = [&](const arma::vec& q, arma::vec& g, double& g0) {
    if (covmode) { g = q - bvec; g0 = 0.0; return; }
    arma::vec r(n);
    switch (family) {
    case GAUS: r = q - y; break;
    case BINO: for (int i = 0; i < n; ++i) r[i] = sigmoid(q[i]) - y[i]; break;
    case POIS: for (int i = 0; i < n; ++i) r[i] = std::exp(q[i]) - y[i]; break;
    case NEGB: for (int i = 0; i < n; ++i) r[i] = (y[i] + theta) * sigmoid(q[i] - cth) - y[i]; break;
    case COX: {
      cox_prep(q);
      double acc = 0.0;
      for (int k = 0; k < n; ++k) { if (dsort[k] > 0) acc += dsort[k] / Dn[k]; S1[k] = acc; }
      for (int k = 0; k < n; ++k) r[ord[k]] = ws[k] * S1[gend[k]] - dsort[k];
      break; }
    }
    g = X.t() * r / (double)n;
    g0 = icpt ? arma::accu(r) / n : 0.0;
  };
  // Bregman divergence f(b+) - f(z) - grad f(z)'(b+ - z), computed stably
  auto curvature = [&](const arma::vec& qn_, const arma::vec& qz_, const arma::vec& d_) -> double {
    if (covmode) return 0.5 * arma::dot(d_, qn_ - qz_);
    double s = 0.0;
    switch (family) {
    case GAUS: { const arma::vec e = qn_ - qz_; s = 0.5 * arma::dot(e, e); break; }
    case BINO: for (int i = 0; i < n; ++i) s += sp_breg(qz_[i], qn_[i] - qz_[i], 0.0); break;
    case POIS: for (int i = 0; i < n; ++i) s += std::exp(qz_[i]) * em1md(qn_[i] - qz_[i]); break;
    case NEGB: for (int i = 0; i < n; ++i)
        s += (y[i] + theta) * sp_breg(qz_[i], qn_[i] - qz_[i], cth);
      break;
    case COX: {
      cox_prep(qz_);
      double a1 = 0.0, a2 = 0.0;
      for (int k = n - 1; k >= 0; --k) {
        const double dl = qn_[ord[k]] - qz_[ord[k]];
        a1 += ws[k] * std::expm1(dl);
        a2 += ws[k] * em1md(dl);
        S1[k] = a1; S2[k] = a2;
      }
      for (int k = 0; k < n; ++k) {
        if (dsort[k] <= 0) continue;
        const double T = S1[gstart[k]] / Dn[k], V = S2[gstart[k]] / Dn[k];
        s += dsort[k] * (V + log1pmx(T));
      }
      break; }
    }
    return s / n;
  };

  arma::vec q = compute_q(beta, a0), qp = q;
  arma::vec z(p), qz, g(p), v(p), bn(p), qn, d(p);
  int nfit = 0;
  double prev_dev = nulldev;

  for (int l = 0; l < nlam; ++l) {
    Rcpp::checkUserInterrupt();
    const double l1 = lam1[l], l2 = lam2[l];
    double t = 1.0, tprev = 1.0;
    double lamref = l1 + l2;
    if (lamref <= 0.0) lamref = std::sqrt(scale) * 1e-3;
    betap = beta; a0p = a0; qp = q;
    int it = 0;
    for (it = 1; it <= maxit; ++it) {
      if (adapt) L = std::max(0.9 * L, Lfloor);
      const double mom = (tprev - 1.0) / t;
      z = beta + mom * (beta - betap);
      const double z0 = a0 + mom * (a0 - a0p);
      qz = q + mom * (q - qp);
      double g0;
      gradient(qz, g, g0);
      double a0n = z0;
      for (int bt = 0; bt < 100; ++bt) {
        v = z - g / L;
        prox_sfl(v, bn, l1 / L, l2 / L, ss);
        a0n = icpt ? z0 - g0 / L : 0.0;
        qn = compute_q(bn, a0n);
        d = bn - z;
        const double d0 = a0n - z0;
        const double dd = arma::dot(d, d) + d0 * d0;
        if (dd <= 0.0) break;
        const double cv = curvature(qn, qz, d);
        if (std::isfinite(cv) && cv <= 0.5 * L * dd * (1.0 + 1e-10)) break;
        L *= 2.0;
      }
      double gm = L * std::max(arma::abs(z - bn).max(), std::fabs(z0 - a0n));
      double dconv = gm / lamref;
      const double rs = arma::dot(z - bn, bn - beta) + (z0 - a0n) * (a0n - a0);
      betap = beta; a0p = a0; qp = q;
      beta = bn; a0 = a0n; q = qn;
      if (rs > 0.0) { t = 1.0; tprev = 1.0; }
      else { const double tn = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t)); tprev = t; t = tn; }
      if (dconv < thresh) break;
    }
    if (it > maxit) it = maxit;
    betas.col(l) = beta;
    a0s[l] = a0;
    iters[l] = it;
    fs[l] = loss(q, beta);
    devs[l] = 2.0 * n * (fs[l] - fsat);
    nfit = l + 1;
    if (early_stop && nulldev > 0.0 && l > 0) {
      const double dr = 1.0 - devs[l] / nulldev;
      const double dr_prev = 1.0 - prev_dev / nulldev;
      if (dr > 0.999) break;
      if (std::fabs(dr - dr_prev) < 1e-5 && l >= 4) break;
      if (!gaus && devs[l] < 1e-5 * nulldev) break;
    }
    prev_dev = devs[l];
  }
  const int nf = std::max(nfit, 1);
  return List::create(_["beta"] = betas.cols(0, nf - 1),
                      _["a0"] = a0s.head(nf),
                      _["iter"] = iters[Range(0, nf - 1)],
                      _["f"] = fs[Range(0, nf - 1)],
                      _["dev"] = devs[Range(0, nf - 1)],
                      _["nfit"] = nfit, _["L"] = L);
}

// Loss value and derivative vector r (d(n*loss)/d eta) for a given linear predictor;
// used on the R side for lambda_max, KKT checks and deviance.
// [[Rcpp::export(.family_eval)]]
List family_eval(const arma::vec& eta, const arma::vec& y, int family, double theta,
                 IntegerVector ord, IntegerVector gstart, IntegerVector gend) {
  const int n = eta.n_elem;
  arma::vec r(n);
  double s = 0.0;
  const double cth = (family == 3) ? std::log(theta) : 0.0;
  switch (family) {
  case 0: for (int i = 0; i < n; ++i) { r[i] = eta[i] - y[i]; s += 0.5 * r[i] * r[i]; } break;
  case 1: for (int i = 0; i < n; ++i) { r[i] = sigmoid(eta[i]) - y[i]; s += softplus(eta[i]) - y[i] * eta[i]; } break;
  case 2: for (int i = 0; i < n; ++i) { const double mu = std::exp(eta[i]); r[i] = mu - y[i]; s += mu - y[i] * eta[i]; } break;
  case 3: for (int i = 0; i < n; ++i) {
      r[i] = (y[i] + theta) * sigmoid(eta[i] - cth) - y[i];
      s += (y[i] + theta) * (cth + softplus(eta[i] - cth)) - y[i] * eta[i];
    } break;
  case 4: {
    const double m = eta.max();
    arma::vec ws(n), Sx(n), Dn(n), C(n);
    for (int k = 0; k < n; ++k) ws[k] = std::exp(eta[ord[k]] - m);
    double acc = 0.0;
    for (int k = n - 1; k >= 0; --k) { acc += ws[k]; Sx[k] = acc; }
    for (int k = 0; k < n; ++k) Dn[k] = Sx[gstart[k]];
    acc = 0.0;
    for (int k = 0; k < n; ++k) {
      const double dk = y[ord[k]];
      if (dk > 0) { acc += dk / Dn[k]; s -= dk * (eta[ord[k]] - m - std::log(Dn[k])); }
      C[k] = acc;
    }
    for (int k = 0; k < n; ++k) r[ord[k]] = ws[k] * C[gend[k]] - y[ord[k]];
    break; }
  }
  return List::create(_["loss"] = s / n, _["r"] = r);
}
