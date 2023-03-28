/** @file linear_elasticity_expressions.cpp

    @brief Linear elasticity problem with adjoint approach for sensitivity
   analysis

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <gismo.h>

using namespace gismo;

namespace gismo::expr {

/*
Expression for the Frobenius matrix product between a matrix of matrices and a
single matrix with cardinality 1
*/
template <typename E1, typename E2>
class frobenius_prod_expr : public _expr<frobenius_prod_expr<E1, E2>> {
 public:
  typedef typename E1::Scalar Scalar;
  enum {
    ScalarValued = 0,
    Space = E1::Space,
    ColBlocks = 0  // E1::ColBlocks || E2::ColBlocks
  };

 private:
  typename E1::Nested_t _u;
  typename E2::Nested_t _v;

  mutable gsMatrix<Scalar> res;

 public:
  frobenius_prod_expr(_expr<E1> const& u, _expr<E2> const& v) : _u(u), _v(v) {
    // todo: add check() functions, which will evaluate expressions on an empty
    // matrix (no points) to setup initial dimensions ???
    GISMO_ASSERT(_u.rows() == _v.rows(), "Wrong dimensions "
                                             << _u.rows() << "!=" << _v.rows()
                                             << " in % operation");
    GISMO_ASSERT(_u.cols() == _v.cols(), "Wrong dimensions "
                                             << _u.cols() << "!=" << _v.cols()
                                             << " in %operation");
  }

  const gsMatrix<Scalar>& eval(const index_t k) const {
    // Evaluate Expressions and cardinality
    const index_t u_r = _u.rows();
    const index_t u_c = _u.cols();

    // Cardinality impl refers to the cols im a matrix
    auto A = _u.eval(k);
    auto B = _v.eval(k);
    const index_t A_rows = A.rows() / u_r;
    const index_t A_cols = A.cols() / u_c;
    GISMO_ASSERT(
        _v.cardinality() == 1,
        "Expression is only for second expressions with cardinality 1");
    GISMO_ASSERT((u_r == _v.cols()) && (u_c == _v.cols()),
                 "Both expressions need to be same size");
    GISMO_ASSERT(B.size() == _v.rows() * _v.cols(),
                 "RHS expression contains more than one matrix");
    res.resize(A_rows, A_cols);
    for (index_t i = 0; i < A_rows; ++i)
      for (index_t j = 0; j < A_cols; ++j)
        res(i, j) =
            (A.block(i * u_r, j * u_c, u_r, u_c).array() * B.array()).sum();
    return res;
  }

  index_t rows() const { return 1; }
  index_t cols() const { return 1; }

  void parse(gsExprHelper<Scalar>& evList) const {
    _u.parse(evList);
    _v.parse(evList);
  }

  const gsFeSpace<Scalar>& rowVar() const { return _u.rowVar(); }
  const gsFeSpace<Scalar>& colVar() const { return _u.colVar(); }  // overwrite

  void print(std::ostream& os) const {
    os << "(";
    _u.print(os);
    os << " % ";
    _v.print(os);
    os << ")";
  }
};
template <typename E1, typename E2>
EIGEN_STRONG_INLINE frobenius_prod_expr<E1, E2> const frobenius(
    _expr<E1> const& u, _expr<E2> const& v) {
  return frobenius_prod_expr<E1, E2>(u, v);
}

}  // namespace gismo::expr

// Global Typedefs
typedef gsExprAssembler<>::geometryMap geometryMap;
typedef gsExprAssembler<>::variable variable;
typedef gsExprAssembler<>::space space;
typedef gsExprAssembler<>::solution solution;

gsMatrix<> GetParameterSensitivities(
    std::string filename,          // Filename for parametrization
    const int& num_refine,         // uniform refinement steps
    const int& dim,                // Geometric dimension
    const gsDofMapper& dof_mapper  // DofMapper of the original problem
) {
  gsFileData<> fd(filename);
  gsMultiPatch<> mp;
  fd.getId(0, mp);
  const int design_dimension = mp.geoDim() / dim;
  gsInfo << "Parameter dimensionality " << design_dimension << std::endl;

  // h-refine each basis
  for (int r = 0; r < num_refine; ++r) {
    mp.uniformRefine();
  }

  // Start the assignement
  gsMatrix<> sensitivities;
  const size_t totalSz = dof_mapper.freeSize();
  sensitivities.resize(totalSz, design_dimension);
  for (size_t j_patch = 0; j_patch != mp.nPatches(); j_patch++) {
    for (index_t k_dim = 0; k_dim != dim; k_dim++) {
      for (size_t l_dof = 0; l_dof != dof_mapper.patchSize(j_patch, k_dim);
           l_dof++) {
        if (dof_mapper.is_free(l_dof, j_patch, k_dim)) {
          const int global_id = dof_mapper.index(l_dof, j_patch, k_dim);
          for (int i_design{}; i_design < design_dimension; i_design++) {
            sensitivities(global_id, i_design) =
                mp.patch(j_patch).coef(l_dof, k_dim + i_design * dim);
          };
        }
      }
    }
  }
  return sensitivities;
}

int main(int argc, char* argv[]) {
  ////////////////////
  // Global Options //
  ////////////////////
  constexpr const int solution_field_dimension{2};

  ////////////////////////////////
  // Parse Command Line Options //
  ////////////////////////////////
  // Title
  gsCmdLine cmd("Linear Elasticity using expressions to prepare Sensitivities");

  // Provide vtk data
  bool plot = false;
  cmd.addSwitch("plot",
                "Create a ParaView visualization file with the solution", plot);
  bool export_xml = false;
  cmd.addSwitch("export-xml", "Export solution into g+smo xml format.",
                export_xml);
  int sample_rate{4};
  cmd.addInt("q", "sample-rate", "Sample rate of splines for export",
             sample_rate);

  // Lame constants
  real_t lame_lambda{2000000}, lame_mu{500000}, rho{1000};
  cmd.addReal("L", "firstLame", "First Lame constant, material parameter",
              lame_lambda);
  cmd.addReal("M", "secondLame", "Second Lame constant, material parameter",
              lame_mu);
  cmd.addReal("R", "rho", "Density", rho);

  // Mesh options
  index_t numRefine = 0;
  cmd.addInt("r", "uniformRefine", "Number of Uniform h-refinement loops",
             numRefine);
  // index_t numElevate = 0;
  // cmd.addInt("e", "degreeElevation",
  //            "Number of degree elevation steps to perform before solving (0:
  //            " "equalize degree in all directions)", numElevate);

  std::string fn("../playground/linear_elasticity/askew_rectangle_mesh.xml");
  cmd.addString("f", "file", "Input XML file", fn);

  std::string fn_param("");
  cmd.addString("x", "parametrization-file", "Control Point sensitivitiy file",
                fn_param);

  // Problem setup
  bool compute_objective_function{false};
  cmd.addSwitch(
      "compute-objective-function",
      "Compute objective function with respect to a given target distribution",
      compute_objective_function);
  bool compute_sensitivities{false};
  cmd.addSwitch(
      "compute-sensitivities",
      "Compute sensitivities with respect to a given objective distribution",
      compute_sensitivities);

  // A few more mesh options
  int mp_id{0}, source_id{1}, bc_id{2}, ass_opt_id{3};
  cmd.addInt("m", "multipach_id", "ID of the multipatch mesh in mesh file",
             mp_id);
  cmd.addInt("s", "source_id", "ID of the source term function in mesh file",
             source_id);
  cmd.addInt("b", "boundary_id",
             "ID of the boundary condition function in mesh file", bc_id);
  cmd.addInt("a", "assembly_options_id",
             "ID of the assembler options in mesh file", ass_opt_id);
#ifdef _OPENMP
  int n_omp_threads{1};
  cmd.addInt("p", "n_threads", "Number of threads used", n_omp_threads);
#endif

  // Parse command line options
  try {
    cmd.getValues(argc, argv);
  } catch (int rv) {
    return rv;
  }

  // Import mesh and load relevant information
  gsFileData<> fd(fn);
  gsInfo << "Loaded file " << fd.lastPath() << "\n";
  gsMultiPatch<> mp;
  fd.getId(mp_id, mp);

  gsFunctionExpr<> f;
  fd.getId(source_id, f);
  gsInfo << "Source function " << f << "\n";
  gsBoundaryConditions<> bc;
  fd.getId(bc_id, bc);
  bc.setGeoMap(mp);
  gsInfo << "Boundary conditions:\n" << bc << "\n";
  gsOptionList Aopt;
  fd.getId(ass_opt_id, Aopt);

  gsInfo << "Geometric dimension " << mp.geoDim() << std::endl;

  //! [Refinement]
  gsMultiBasis<> function_basis(mp, true);  // true: poly-splines (not NURBS)

  // h-refine each basis
  for (int r = 0; r < numRefine; ++r) {
    function_basis.uniformRefine();
  }

  // Output user information
  gsInfo << "Patches: " << mp.nPatches()
         << ", min-degree: " << function_basis.minCwiseDegree()
         << ", min-degree: " << function_basis.maxCwiseDegree() << "\n";
#ifdef _OPENMP
  gsInfo << "Available threads: " << omp_get_max_threads() << "\n";
  omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
  gsInfo << "Number of threads: " << omp_get_num_threads() << "\n";
#endif

  ///////////////////
  // Problem Setup //
  ///////////////////

  // Expression assembler
  gsExprAssembler<> expr_assembler(1, 1);
  expr_assembler.setOptions(Aopt);
  gsInfo << "Active options:\n" << expr_assembler.options() << "\n";

  // Elements used for numerical integration
  expr_assembler.setIntegrationElements(function_basis);

  // Set the geometry map
  geometryMap geom_expr = expr_assembler.getMap(mp);

  // Set the discretization space
  space u_trial =
      expr_assembler.getSpace(function_basis, solution_field_dimension);

  // Set the source term
  auto ff = expr_assembler.getCoeff(f, geom_expr);

  // Solution vector and solution variable
  gsMatrix<> solVector;
  solution solution_expression = expr_assembler.getSolution(u_trial, solVector);

  // Setup values for timing
  double setup_time(0), ma_time(0), slv_time(0);
  gsStopwatch timer;

  u_trial.setup(bc, dirichlet::l2Projection, 0);

  // Initialize the system
  expr_assembler.initSystem();
  setup_time += timer.stop();

  gsInfo << "Number of degrees of freedom : " << expr_assembler.numDofs()
         << std::endl;

  //////////////
  // Assembly //
  //////////////
  gsInfo << "Starting assembly of linear system ..." << std::flush;
  timer.restart();

  // Compute the system matrix and right-hand side
  auto phys_jacobian = ijac(u_trial, geom_expr);
  auto bilin_lambda = lame_lambda * idiv(u_trial, geom_expr) *
                      idiv(u_trial, geom_expr).tr() * meas(geom_expr);
  auto bilin_mu_1 = lame_mu * (phys_jacobian.cwisetr() % phys_jacobian.tr()) *
                    meas(geom_expr);
  auto bilin_mu_2 =
      lame_mu * (phys_jacobian % phys_jacobian.tr()) * meas(geom_expr);
  auto lin_form = rho * u_trial * ff * meas(geom_expr);

  auto bilin_combined = (bilin_lambda + bilin_mu_1 + bilin_mu_2);

  expr_assembler.assemble(bilin_combined);
  expr_assembler.assemble(lin_form);

  // Compute the Neumann terms defined on physical space
  auto g_N = expr_assembler.getBdrFunction(geom_expr);
  // Neumann conditions seem broken here
  expr_assembler.assembleBdr(bc.get("Neumann"),
                             u_trial * g_N * nv(geom_expr).norm());

  ma_time += timer.stop();
  gsInfo << "\t\tFinished" << std::endl;

  ///////////////////
  // Linear Solver //
  ///////////////////
  gsInfo << "Solving the linear system of equations ..." << std::flush;
  timer.restart();
  const auto& matrix_in_initial_configuration = expr_assembler.matrix();
  const auto rhs_vector = expr_assembler.rhs();

  // Initialize linear solver
  gsSparseSolver<>::CGDiagonal solver;
  solver.compute(matrix_in_initial_configuration);
  solVector = solver.solve(expr_assembler.rhs());
  slv_time += timer.stop();
  gsInfo << "\t\t\tFinished" << std::endl;

  //////////////////////////////
  // Export and Visualization //
  //////////////////////////////
  gsExprEvaluator<> expression_evaluator(expr_assembler);

  // Generate Paraview File
  gsInfo << "Starting the paraview export ..." << std::flush;
  if (plot) {
    gsParaviewCollection collection("ParaviewOutput/solution",
                                    &expression_evaluator);
    collection.options().setSwitch("plotElements", true);
    collection.options().setInt("plotElements.resolution", sample_rate);
    collection.newTimeStep(&mp);
    collection.addField(solution_expression, "numerical solution");
    collection.saveTimeStep();
    collection.save();

  } else {
    gsInfo << "skipping";
  }
  gsInfo << "\tFinished" << std::endl;
  //! [Export visualization in ParaView]

  // Export solution file as xml
  gsInfo << "Starting the xml export ..." << std::flush;
  if (export_xml) {
    gsMultiPatch<> mpsol;
    gsMatrix<> full_solution;
    gsFileData<> output;
    output << solVector;
    solution_expression.extractFull(full_solution);
    output << full_solution;
    output.save("solution-field.xml");
  } else {
    gsInfo << "skipping";
  }
  gsInfo << "\tFinished" << std::endl;

  ////////////////////////////////////////////////////
  // Optimization Requirements - Objective Function //
  ////////////////////////////////////////////////////
  if (compute_objective_function) {
    gsInfo << "Computing the objective function value ..." << std::flush;
    // Norm of displacement in Neumann region
    expr_assembler.clearRhs();
    space u_trial_single_var = expr_assembler.getSpace(function_basis, 1);
    // Assemble with test function and using the sum over all integrals (using
    // partition of unity), this is a bit inefficient but only on a subdomain
    expr_assembler.assembleBdr(
        bc.get("Neumann"),
        u_trial_single_var * (solution_expression.tr() * solution_expression) *
            nv(geom_expr).norm());
    const auto objective_function_value = expr_assembler.rhs().sum();
    gsInfo << "\tFinished Vaule : " << std::setprecision(20)
           << objective_function_value << std::endl;

    // Assemble derivatives of objective function with respect to field
    if (compute_sensitivities) {
      //////////////////////////////////////
      // Derivative of Objective Function //
      //////////////////////////////////////
      gsInfo << "Computing the objective function derivative ..." << std::flush;
      expr_assembler.clearRhs();
      expr_assembler.assembleBdr(
          bc.get("Neumann"),
          2 * u_trial * solution_expression * nv(geom_expr).norm());
      const auto objective_function_derivative = expr_assembler.rhs();
      gsInfo << "\tFinished" << std::endl;

      /////////////////////////////////
      // Solving the adjoint problem //
      /////////////////////////////////
      gsInfo << "Solving the adjoint equation ..." << std::flush;
      timer.restart();
      const gsSparseMatrix<> matrix_in_initial_configuration(
          expr_assembler.matrix().transpose().eval());
      auto rhs_vector = expr_assembler.rhs();

      // Initialize linear solver
      gsSparseSolver<>::CGDiagonal solverAdjoint;
      gsMatrix<> lagrange_multipliers;
      solverAdjoint.compute(matrix_in_initial_configuration);
      lagrange_multipliers = -solverAdjoint.solve(expr_assembler.rhs());
      slv_time += timer.stop();

      gsInfo << "\t\tFinished - T : " << slv_time << std::endl;

      ////////////////////////////////
      // Derivative of the LHS Form //
      ////////////////////////////////
      expr_assembler.clearRhs();
      expr_assembler.clearMatrix();

      // Auxiliary expressions
      auto jacobian = jac(geom_expr);                      // validated
      auto inv_jacs = jacobian.ginv();                     // validated
      auto meas_expr = meas(geom_expr);                    // validated
      auto djacdc = jac(u_trial);                          // validated
      auto aux_expr = (djacdc * inv_jacs).tr();            // validated
      auto meas_expr_dx = meas_expr * (aux_expr).trace();  // validated

      // Start to assemble the bilinear form with the known solution field
      // 1. Bilinear form of lambda expression seperated into 3 individual
      // sections
      auto BL_lambda_1 =
          idiv(solution_expression, geom_expr).val();  // validated
      auto BL_lambda_2 = idiv(u_trial, geom_expr);     // validated
      auto BL_lambda =
          lame_lambda * BL_lambda_2 * BL_lambda_1 * meas_expr;  // validated

      // trace(A * B) = A:B^T
      auto BL_lambda_1_dx = frobenius(
          aux_expr, ijac(solution_expression, geom_expr));          // validated
      auto BL_lambda_2_dx = (ijac(u_trial, geom_expr) % aux_expr);  // validated

      auto BL_lambda_dx =
          lame_lambda * BL_lambda_2 * BL_lambda_1 * meas_expr_dx -
          lame_lambda * BL_lambda_2_dx * BL_lambda_1 * meas_expr -
          lame_lambda * BL_lambda_2 * BL_lambda_1_dx * meas_expr;  // validated

      // Assemble
      expr_assembler.assemble(BL_lambda_dx);

      // 2. Bilinear form of mu (first part)
      // BL_mu1_2 seems to be in a weird order with [jac0, jac2] leading
      // to [2x(2nctps)]
      auto BL_mu1_1 = ijac(solution_expression, geom_expr);       // validated
      auto BL_mu1_2 = ijac(u_trial, geom_expr);                   // validated
      auto BL_mu1 = lame_mu * (BL_mu1_2 % BL_mu1_1) * meas_expr;  // validated

      auto BL_mu1_1_dx = -(ijac(solution_expression, geom_expr) *
                           aux_expr.cwisetr());  //          validated
      auto BL_mu1_2_dx =
          -(jac(u_trial) * inv_jacs * aux_expr.cwisetr());  // validated

      auto BL_mu1_dx0 =
          lame_mu * BL_mu1_2 % BL_mu1_1_dx * meas_expr;  // validated
      auto BL_mu1_dx1 =
          lame_mu * frobenius(BL_mu1_2_dx, BL_mu1_1) * meas_expr;  // validated
      auto BL_mu1_dx2 = lame_mu * frobenius(BL_mu1_2, BL_mu1_1).cwisetr() *
                        meas_expr_dx;  // validated

      // Assemble
      expr_assembler.assemble(BL_mu1_dx0);
      expr_assembler.assemble(BL_mu1_dx1);
      expr_assembler.assemble(BL_mu1_dx2);

      // 2. Bilinear form of mu (first part)
      auto BL_mu2_1 =
          ijac(solution_expression, geom_expr).cwisetr();         // validated
      auto& BL_mu2_2 = BL_mu1_2;                                  // validated
      auto BL_mu2 = lame_mu * (BL_mu2_2 % BL_mu2_1) * meas_expr;  // validated

      auto inv_jac_T = inv_jacs.tr();
      auto BL_mu2_1_dx = -inv_jac_T * jac(u_trial).tr() * inv_jac_T *
                         jac(solution_expression).cwisetr();  // validated
      auto& BL_mu2_2_dx = BL_mu1_2_dx;                        // validated

      auto BL_mu2_dx0 =
          lame_mu * BL_mu2_2 % BL_mu2_1_dx * meas_expr;  // validated
      auto BL_mu2_dx1 =
          lame_mu * frobenius(BL_mu2_2_dx, BL_mu2_1) * meas_expr;  // validated
      auto BL_mu2_dx2 = lame_mu * frobenius(BL_mu2_2, BL_mu2_1).cwisetr() *
                        meas_expr_dx;  // validated

      // Assemble
      expr_assembler.assemble(BL_mu2_dx0);
      expr_assembler.assemble(BL_mu2_dx1);
      expr_assembler.assemble(BL_mu2_dx2);

      // Linear Form Part
      auto LF_1 = -rho * u_trial * ff * meas_expr;
      auto LF_1_dx = -rho * u_trial * ff * meas_expr_dx;

      // Assemble
      expr_assembler.assemble(LF_1_dx);

      ///////////////////////////
      // Compute sensitivities //
      ///////////////////////////
      const auto sensitivities =
          lagrange_multipliers.transpose() * expr_assembler.matrix();
      gsDofMapper dof_mapper = u_trial.mapper();

      const gsMatrix<>& parameter_sensitivity = GetParameterSensitivities(
          fn_param, numRefine, mp.geoDim(), dof_mapper);

      // Check DEBUG
      gsInfo << "Sensitivity is " << sensitivities * parameter_sensitivity
             << std::endl;
    }
  }

  // User output infor timings
  gsInfo << "\n\nTotal time: " << setup_time + ma_time + slv_time << "\n";
  gsInfo << "     Setup: " << setup_time << "\n";
  gsInfo << "  Assembly: " << ma_time << "\n";
  gsInfo << "   Solving: " << slv_time << "\n" << std::flush;
  return EXIT_SUCCESS;

}  // end main
