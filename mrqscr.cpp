#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List mrqscr_cpp(const arma::vec& y, 
                            const arma::mat& x_mat, 
                            const arma::vec& tau_vec,
                            Rcpp::Nullable<arma::mat> z_mat = R_NilValue,
                            int order = 3,
                            bool interaction = false) {
  int n = y.n_elem;
  int p_init = x_mat.n_cols;
  int p = 2;
  
  if (z_mat.isNotNull()) {
    arma::mat z = Rcpp::as<arma::mat>(z_mat);
    p = interaction ? 2 + 2 * z.n_cols : 2 + z.n_cols;
  }
  
  int L = tau_vec.n_elem;
  int k = order - 1;
  int N = L + k;
  
  // Generate knots
  arma::vec knots = arma::linspace<arma::vec>(arma::min(tau_vec), arma::max(tau_vec), L + 1);
  // Rcpp::Rcout << "Knots: " << knots.t() << std::endl;
  
  // Call bsplineS from fda package
  Rcpp::Environment fda_env = Rcpp::Environment::namespace_env("fda");
  Rcpp::Function bsplineS = fda_env["bsplineS"];
  Rcpp::NumericMatrix spline_basis = bsplineS(tau_vec, knots, Rcpp::Named("norder") = k + 1);
  // Rcpp::Rcout << "Spline Basis: " << Rcpp::as<arma::mat>(spline_basis) << std::endl;
  
  arma::mat v_star = arma::zeros(L * n, N * p);
  arma::vec result_coef = arma::zeros(p_init);
  arma::vec result_coef_ori = arma::zeros(p_init);
  
  for (int iter = 0; iter < p_init; ++iter) {
    // Rcpp::Rcout << "Processing iteration: " << iter + 1 << " of " << p_init << std::endl;
    // Rcpp::Rcout << "Iteration: " << iter + 1 << std::endl;
    
    // Define X for each iteration
    arma::mat x;
    if (z_mat.isNull()) {
      x = arma::join_horiz(arma::ones(n), x_mat.col(iter)); // Add intercept
    } else {
      arma::mat z = Rcpp::as<arma::mat>(z_mat);
      if (interaction) {
        x = arma::join_horiz(arma::ones(n), x_mat.col(iter), z, x_mat.col(iter) % z);
      } else {
        x = arma::join_horiz(arma::ones(n), x_mat.col(iter), z);
      }
    }
    // Rcpp::Rcout << "X Matrix: " << x << std::endl;
    
    // Convert spline_basis to Armadillo matrix
    arma::mat arma_spline_basis = Rcpp::as<arma::mat>(spline_basis);
    
    // Populate v_star using Kronecker products
    for (int j = 0; j < L; ++j) {
      for (int i = 0; i < n; ++i) {
        arma::rowvec row_spline = arma_spline_basis.row(j);
        // Rcpp::Rcout << "C++: row_spline (j=" << j << "): " << row_spline.t() << std::endl;
        
        arma::mat kron_result = arma::kron(row_spline, x.row(i).t()); // Kronecker product
        arma::rowvec flattened_kron_result = arma::vectorise(kron_result.t(), 1); // Transpose then flatten
        
        v_star.row(j * n + i) = flattened_kron_result;
        // Rcpp::Rcout << "C++: v_star row (" << j * n + i << "): " << v_star.row(j * n + i) << std::endl;
      }
    }
    
    // Rcpp::Rcout << "v_star: " << v_star << std::endl;
    
    // Create constraints
    arma::mat A = arma::join_horiz(v_star, -v_star, arma::eye(L * n, L * n), -arma::eye(L * n, L * n));
    arma::vec b_star = arma::repmat(y, L, 1);
    arma::vec c = arma::join_vert(
      arma::zeros(2 * p * N),
      arma::vectorise(arma::kron(tau_vec, arma::ones(n)), 0),
      arma::vectorise(arma::kron((arma::ones(L) - tau_vec), arma::ones(n)), 0)
    );
    // Rcpp::Rcout << "A Matrix: " << A << std::endl;
    // Rcpp::Rcout << "b_star: " << b_star.t() << std::endl;
    // Rcpp::Rcout << "c: " << c.t() << std::endl;
    
    // Call lp in R
    Rcpp::Environment lpSolve_env = Rcpp::Environment::namespace_env("lpSolve");
    Rcpp::Function lp = lpSolve_env["lp"];
    Rcpp::List linprog = lp("min", c, A, Rcpp::CharacterVector(L * n, "="), b_star);
    
    arma::vec solution = Rcpp::as<arma::vec>(linprog["solution"]);
    arma::mat est_theta = arma::reshape(
      solution.subvec(0, p * N - 1) - solution.subvec(p * N, 2 * p * N - 1),
      p,
      N
    );
    // Rcpp::Rcout << "est_theta: " << est_theta << std::endl;
    
    arma::mat alpha_tau = est_theta * Rcpp::as<arma::mat>(spline_basis).t();
    // Rcpp::Rcout << "alpha_tau: " << alpha_tau << std::endl;
    
    result_coef(iter) = std::sqrt(arma::accu(arma::square(alpha_tau.row(1))));
    result_coef_ori(iter) = arma::accu(arma::sign(alpha_tau.row(1)));
  }
  
  // Final result
  // Rcpp::Rcout << "Result Coefficients: " << result_coef.t() << std::endl;
  
  // arma::uvec ranks = arma::sort_index(-result_coef);
  
  // Compute ranks for result_coef
  arma::uvec sorted_indices = arma::sort_index(-result_coef);  // Indices for descending order
  arma::vec rank = arma::zeros<arma::vec>(p_init);
  
  // Assign ranks based on sorted indices
  for (arma::uword i = 0; i < sorted_indices.n_elem; ++i) {
    rank(sorted_indices(i)) = i + 1;  // 1-based indexing for ranks
  }
  
  return Rcpp::List::create(
    Rcpp::Named("rank") = Rcpp::wrap(arma::conv_to<std::vector<double>>::from(rank)),  // Convert to R vector
    Rcpp::Named("coef") = Rcpp::wrap(arma::conv_to<std::vector<double>>::from(result_coef)),  // Convert to R vector
    Rcpp::Named("sign") = Rcpp::wrap(arma::conv_to<std::vector<double>>::from(result_coef_ori))  // Convert to R vector
  );
}

// [[Rcpp::export]]
Rcpp::List mrq_cpp_intercept(const arma::vec& y, 
                             const arma::vec& tau_vec,
                             int order = 3,
                             bool interaction = false) {
  int n = y.n_elem;
  
  // Build design matrix for intercept-only model: an n x 1 column of ones.
  arma::mat x = arma::ones(n, 1);
  int p = x.n_cols;  // p = 1
  
  // Set up spline and LP parameters:
  int L = tau_vec.n_elem;
  int k = order - 1;
  int N = L + k;
  
  // Generate knots between the min and max of tau_vec.
  arma::vec knots = arma::linspace<arma::vec>(arma::min(tau_vec), arma::max(tau_vec), L + 1);
  
  // Get the B-spline basis from the 'fda' package:
  Rcpp::Environment fda_env = Rcpp::Environment::namespace_env("fda");
  Rcpp::Function bsplineS = fda_env["bsplineS"];
  Rcpp::NumericMatrix spline_basis = bsplineS(tau_vec, knots, Rcpp::Named("norder") = k + 1);
  arma::mat arma_spline_basis = Rcpp::as<arma::mat>(spline_basis);
  
  // Build v_star matrix, dimensions: (L*n) x (N*p). With p = 1, the dimension is (L*n) x N.
  arma::mat v_star = arma::zeros(L * n, N * p);
  for (int j = 0; j < L; ++j) {
    for (int i = 0; i < n; ++i) {
      arma::rowvec row_spline = arma_spline_basis.row(j);
      // For intercept-only, x.row(i).t() is [1], so the Kronecker product returns row_spline.
      arma::mat kron_result = arma::kron(row_spline, x.row(i).t());
      // Flatten the transposed result to get a row vector of length N.
      arma::rowvec flattened = arma::vectorise(kron_result.t(), 1);
      v_star.row(j * n + i) = flattened;
    }
  }
  
  // Construct the LP constraint matrix and vectors.
  // The LP variables are ordered as (θ⁺, θ⁻, u⁺, u⁻).
  arma::mat A = arma::join_horiz(v_star, -v_star, arma::eye(L * n, L * n), -arma::eye(L * n, L * n));
  arma::vec b_star = arma::repmat(y, L, 1);
  
  arma::vec term1 = arma::vectorise(arma::kron(tau_vec, arma::ones(n)), 0);
  arma::vec term2 = arma::vectorise(arma::kron((arma::ones(L) - tau_vec), arma::ones(n)), 0);
  arma::vec c = arma::join_vert(arma::zeros(2 * p * N), term1, term2);
  
  // Create a CharacterVector of equality signs for each constraint row.
  Rcpp::CharacterVector eqConstraints(L * n);
  for (int i = 0; i < L * n; i++) {
    eqConstraints[i] = "=";
  }
  
  // Solve the LP problem using lpSolve from R.
  Rcpp::Environment lpSolve_env = Rcpp::Environment::namespace_env("lpSolve");
  Rcpp::Function lp = lpSolve_env["lp"];
  Rcpp::List linprog = lp("min", c, A, eqConstraints, b_star);
  
  arma::vec solution = Rcpp::as<arma::vec>(linprog["solution"]);
  
  // Recover theta by subtracting the negative part from the positive part,
  // then reshape to a (p x N) matrix. (Here, p = 1.)
  arma::mat est_theta = arma::reshape(
    solution.subvec(0, p * N - 1) - solution.subvec(p * N, 2 * p * N - 1),
    p,
    N
  );
  
  // Multiply by the transposed spline basis to obtain the coefficient function over tau.
  // This yields a (p x L) matrix; with p = 1, a 1 x L vector.
  arma::mat alpha_tau = est_theta * arma_spline_basis.t();
  
  // Return the coefficient matrix.
  return Rcpp::List::create(Rcpp::Named("coef") = alpha_tau);
}



// [[Rcpp::export]]
Rcpp::List mrq_cpp_full(const arma::vec& y, 
                        const arma::mat& x_mat, 
                        const arma::vec& tau_vec,
                        Rcpp::Nullable<arma::mat> z_mat = R_NilValue,
                        int order = 3,
                        bool interaction = false) {
  int n = y.n_elem;
  int p_init = x_mat.n_cols;
  arma::mat x;
  
  // Build full design matrix including intercept and possibly z_mat and interactions
  if (z_mat.isNull()) {
    // No extra covariate: include intercept and all columns of x_mat
    x = arma::join_horiz(arma::ones(n), x_mat);
  } else {
    arma::mat z = Rcpp::as<arma::mat>(z_mat);
    if (interaction) {
      // Include interactions between each column of x_mat and each column of z
      arma::mat inter = arma::zeros(n, p_init * z.n_cols);
      for (int i = 0; i < p_init; ++i) {
        for (arma::uword j = 0; j < z.n_cols; ++j) {
          inter.col(i * z.n_cols + j) = x_mat.col(i) % z.col(j);
        }
      }
      x = arma::join_horiz(arma::ones(n), x_mat, z, inter);
    } else {
      // Without interactions, simply combine the matrices
      x = arma::join_horiz(arma::ones(n), x_mat, z);
    }
  }
  
  // Get effective number of predictors from the full design matrix
  int p = x.n_cols;
  
  // Set up spline and LP parameters
  int L = tau_vec.n_elem;
  int k = order - 1;
  int N = L + k;
  
  // Generate knots between min and max of tau_vec
  arma::vec knots = arma::linspace<arma::vec>(arma::min(tau_vec), arma::max(tau_vec), L + 1);
  
  // Get B-spline basis from the 'fda' package
  Rcpp::Environment fda_env = Rcpp::Environment::namespace_env("fda");
  Rcpp::Function bsplineS = fda_env["bsplineS"];
  Rcpp::NumericMatrix spline_basis = bsplineS(tau_vec, knots, Rcpp::Named("norder") = k + 1);
  arma::mat arma_spline_basis = Rcpp::as<arma::mat>(spline_basis);
  
  // Build the v_star matrix (dimension: (L*n) x (N*p))
  arma::mat v_star = arma::zeros(L * n, N * p);
  for (int j = 0; j < L; ++j) {
    for (int i = 0; i < n; ++i) {
      arma::rowvec row_spline = arma_spline_basis.row(j);
      // Compute Kronecker product: (1 x N) kron (p x 1) gives a (p x N) matrix
      arma::mat kron_result = arma::kron(row_spline, x.row(i).t());
      // Flatten the transpose so that the result is a row vector of length p*N
      arma::rowvec flattened = arma::vectorise(kron_result.t(), 1);
      v_star.row(j * n + i) = flattened;
    }
  }
  
  // Construct the linear programming (LP) constraint matrix and vectors
  // The LP constraints are set up for the variables (θ⁺, θ⁻, u⁺, u⁻)
  arma::mat A = arma::join_horiz(v_star, -v_star, arma::eye(L * n, L * n), -arma::eye(L * n, L * n));
  arma::vec b_star = arma::repmat(y, L, 1);
  arma::vec c = arma::join_vert(
    arma::zeros(2 * p * N),
    arma::vectorise(arma::kron(tau_vec, arma::ones(n)), 0),
    arma::vectorise(arma::kron((arma::ones(L) - tau_vec), arma::ones(n)), 0)
  );
  
  // Solve the LP problem using lpSolve in R
  Rcpp::Environment lpSolve_env = Rcpp::Environment::namespace_env("lpSolve");
  Rcpp::Function lp = lpSolve_env["lp"];
  Rcpp::List linprog = lp("min", c, A, Rcpp::CharacterVector(L * n, "="), b_star);
  
  arma::vec solution = Rcpp::as<arma::vec>(linprog["solution"]);
  // The solution is split into positive and negative parts for theta.
  // We subtract the second half from the first, and then reshape to get the theta matrix.
  arma::mat est_theta = arma::reshape(
    solution.subvec(0, p * N - 1) - solution.subvec(p * N, 2 * p * N - 1),
    p,
    N
  );
  
  // Multiply by the transposed spline basis to get the coefficient curves,
  // yielding a matrix of size (p x L)
  arma::mat alpha_tau = est_theta * arma_spline_basis.t();
  
  // Return just the coefficient matrix as requested.
  return Rcpp::List::create(Rcpp::Named("coef") = alpha_tau);
}
