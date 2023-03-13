/** @file linear_elasticity_expressions.cpp

    @brief Linear elasticity problem with adjoint approach for sensitivity
   analysis

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

//! [Include namespace]
#include <gismo.h>

using namespace gismo;

// Global Typedefs
typedef gsExprAssembler<>::geometryMap geometryMap;
typedef gsExprAssembler<>::variable variable;
typedef gsExprAssembler<>::space space;
typedef gsExprAssembler<>::solution solution;

/**
 * @brief Function is only temporary and serves as a FD approximation to
 * validate expressions
 *
 * to be deleted
 */
auto ComputeSensitivityFD(const gsMultiPatch<>& mp_dx, const gsOptionList& Aopt,
                          const gsMultiBasis<>& function_basis,
                          const double& lame_lambda, const double& lame_mu,
                          const gsMatrix<>& solVector_reference,
                          const gsBoundaryConditions<>& bc) {
  gsExprAssembler<> expr_assembler(1, 1);
  expr_assembler.setOptions(Aopt);

  // Elements used for numerical integration
  expr_assembler.setIntegrationElements(function_basis);

  // Set the geometry map
  geometryMap geom_expr = expr_assembler.getMap(mp_dx);

  // Set the discretization space
  space u_trial = expr_assembler.getSpace(function_basis, 2);

  // Solution space

  gsMatrix<> solVector{solVector_reference};
  solution solution_expression = expr_assembler.getSolution(u_trial, solVector);

  u_trial.setup(bc, dirichlet::l2Projection, 0);
  // Compute the system matrix and right-hand side

  // Assemble
  // Auxiliary expressions
  auto meas_expr = meas(geom_expr);
  auto BL_lambda_1 = idiv(solution_expression, geom_expr).val();  // validated
  auto BL_lambda_2 = idiv(u_trial, geom_expr);                    // validated
  auto BL_lambda =
      lame_lambda * BL_lambda_2 * BL_lambda_1 * meas_expr;    // validated
  auto BL_mu1_1 = ijac(solution_expression, geom_expr);       // validated
  auto BL_mu1_2 = ijac(u_trial, geom_expr);                   // validated
  auto BL_mu1 = lame_mu * (BL_mu1_2 % BL_mu1_1) * meas_expr;  // validated
  expr_assembler.initSystem();
  expr_assembler.clearRhs();
  expr_assembler.assemble(BL_lambda);

  // Evaluator for simplified expressions
  gsExprEvaluator<> expression_evaluator(expr_assembler);
  gsMatrix<> evalPoint(2, 1);
  evalPoint << .25, .6;

  gsInfo << "BL_mu1_2 : \n"
         << std::setprecision(20)
         << expression_evaluator.eval(BL_mu1_2, evalPoint) << std::endl;

  // Return the matrix to evaluate the residual
  return expr_assembler.rhs();
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
  index_t numElevate = 0;
  cmd.addInt("e", "degreeElevation",
             "Number of degree elevation steps to perform before solving (0: "
             "equalize degree in all directions)",
             numElevate);

  std::string fn("../playground/linear_elasticity/askew_rectangle_mesh.xml");
  cmd.addString("f", "file", "Input XML file", fn);

  // Testing
  bool fd_test{false};
  double ddx{0.00001};
  cmd.addSwitch("fd-test", "Calculate the fd solution of bilinear form",
                fd_test);
  cmd.addReal("d", "ddx", "Calculate the fd solution of bilinear form", ddx);

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

  ////////////////////////////////
  // Problem Setup and Assembly //
  ////////////////////////////////

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

  gsInfo << expr_assembler.numDofs() << std::endl;

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
  gsInfo << "Combined bilinear form is assembled" << std::endl;
  expr_assembler.assemble(lin_form);
  gsInfo << "Volume part of linear form is assembled" << std::endl;

  // Compute the Neumann terms defined on physical space
  // auto g_N = expr_assembler.getBdrFunction(geom_expr);
  // Neumann conditions seem broken here
  // expr_assembler.assembleBdr(bc.get("Neumann"),
  //                            u_trial * g_N.val() * nv(geom_expr).tr());

  ma_time += timer.stop();
  gsInfo << "Finished Assembly" << std::endl;

  ///////////////////
  // Linear Solver //
  ///////////////////
  timer.restart();
  const auto& matrix_in_initial_configuration = expr_assembler.matrix();
  const auto rhs_vector = expr_assembler.rhs();

  // Initialize linear solver
  gsSparseSolver<>::CGDiagonal solver;
  solver.compute(matrix_in_initial_configuration);
  solVector = solver.solve(expr_assembler.rhs());
  slv_time += timer.stop();
  gsInfo << "Finished solving linear system" << std::endl;

  // User output infor timings
  gsInfo << "\n\nTotal time: " << setup_time + ma_time + slv_time << "\n";
  gsInfo << "     Setup: " << setup_time << "\n";
  gsInfo << "  Assembly: " << ma_time << "\n";
  gsInfo << "   Solving: " << slv_time << "\n" << std::flush;

  //////////////////////////////
  // Export and Visualization //
  //////////////////////////////
  gsExprEvaluator<> expression_evaluator(expr_assembler);

  // Generate Paraview File
  if (plot) {
    gsInfo << "Plotting in Paraview...\n";
    expression_evaluator.options().setSwitch("plot.elements", false);
    expression_evaluator.writeParaview(solution_expression, geom_expr,
                                       "solution");

    gsFileManager::open("solution.pvd");
  } else {
    gsInfo << "Done. No output created, re-run with --plot to get a "
              "ParaView "
              "file containing the solution.\n";
  }
  //! [Export visualization in ParaView]

  // Export solution file as xml
  if (export_xml) {
    gsInfo << "Writing to G+Smo XML." << std::flush;
    gsMultiPatch<> mpsol;
    gsMatrix<> full_solution;

    gsFileData<> output;

    output << solVector;

    solution_expression.extractFull(full_solution);
    output << full_solution;
    output.save("solution-field.xml");
  } else {
    gsInfo << "Export in Paraview format only, no xml output created.\n";
  }
  gsInfo << std::endl;

  // For local evaluation prior to testing
  gsExprEvaluator<> evaluator{expr_assembler};
  gsMatrix<> evalPoint(2, 1);
  evalPoint << .25, .6;

  // Auxiliary for validation of expressions
  auto print_function_expressions = [&](const std::string& name,
                                        auto expression) {
    gsInfo << "\nThe expression " << name << " : " << expression
           << " evaluates at (" << evalPoint(0) << ", " << evalPoint(1)
           << ") to \n"
           << evaluator.eval(expression, evalPoint) << std::endl;
    gsInfo << "It has \t" << expression.rows() << " rows and \t"
           << expression.cols() << " cols" << std::endl;
    gsInfo << "Is Vector : " << expression.isVector()
           << " is Matrix : " << expression.isMatrix() << std::endl;
    gsInfo << "The cardinality of the expression is : "
           << expression.cardinality() << std::endl;
  };

  ////////////////////////////////////////////////////
  // Optimization Requirements - Objective Function //
  ////////////////////////////////////////////////////
  if (compute_objective_function) {
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

    // Assemble derivatives of objective function with respect to field
    if (compute_sensitivities) {
      //////////////////////////////////////
      // Derivative of Objective Function //
      //////////////////////////////////////
      expr_assembler.clearRhs();
      expr_assembler.assembleBdr(
          bc.get("Neumann"),
          2 * u_trial * solution_expression * nv(geom_expr).norm());
      const auto objective_function_derivative = expr_assembler.rhs();

      ////////////////////////////////
      // Derivative of the LHS Form //
      ////////////////////////////////

      // Auxiliary expressions
      auto jacobian = jac(geom_expr);                      // validated
      auto inv_jacs = jacobian.ginv();                     // validated
      auto djacdc = jac(u_trial);                          // validated
      auto aux_expr = (djacdc * inv_jacs).tr();            // validated
      auto meas_expr = meas(geom_expr);                    // validated
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
      auto BL_lambda_1_dx =
          (ijac(solution_expression, geom_expr) * aux_expr.tr())
              .trace()
              .tr();                                                // validated
      auto BL_lambda_2_dx = (ijac(u_trial, geom_expr) % aux_expr);  // validated
      auto BL_lambda_dx =
          lame_lambda * BL_lambda_2 * BL_lambda_1 * meas_expr_dx -
          lame_lambda * BL_lambda_2_dx * BL_lambda_1 * meas_expr -
          lame_lambda * BL_lambda_2 * BL_lambda_1_dx * meas_expr;  // validated

      // 2. Bilinear form of mu (first part)
      auto BL_mu1_1 = ijac(solution_expression, geom_expr);       // validated
      auto BL_mu1_2 = ijac(u_trial, geom_expr);                   // validated
      auto BL_mu1 = lame_mu * (BL_mu1_2 % BL_mu1_1) * meas_expr;  // validated
      auto BL_mu1_1_dx = (ijac(solution_expression, geom_expr) *
                          aux_expr.tr());  //          validated

      auto BL_mu1_2_dx = ijac(u_trial, geom_expr).cwisetr() * aux_expr;  // WIP
      // auto BL_mu1_dx = -lame_mu * (BL_mu1_2 % BL_mu1_1_dx) * meas_expr -
      //                  lame_mu * (BL_mu1_2_dx % BL_mu1_1) * meas_expr +
      //                  lame_mu * (BL_mu1_2 % BL_mu1_1) * meas_expr_dx;

      gsInfo << "djacdc : \n"
             << std::setprecision(20)
             << expression_evaluator.eval(djacdc, evalPoint) << std::endl;

      gsInfo << "djacdc * inv_jacs : \n"
             << std::setprecision(20)
             << expression_evaluator.eval(djacdc * inv_jacs, evalPoint)
             << std::endl;

      gsInfo << "aux_expr : \n"
             << std::setprecision(20)
             << expression_evaluator.eval(aux_expr, evalPoint) << std::endl;

      // print_function_expressions("BL_mu1", BL_mu1);
      // print_function_expressions("BL_mu1_1", BL_mu1_1);
      // print_function_expressions("BL_mu1_1_dx", BL_mu1_1_dx);
      // print_function_expressions("BL_mu1_2", BL_mu1_2);
      // print_function_expressions("BL_mu1_2_dx", BL_mu1_2_dx);
      // print_function_expressions(
      //     "-lame_mu * (BL_mu1_2 % BL_mu1_1_dx) * meas_expr",
      //     -lame_mu * (BL_mu1_2 % BL_mu1_1_dx) * meas_expr);
      // print_function_expressions(
      //     "lame_mu * (BL_mu1_1 % BL_mu1_2).tr()",
      //     lame_mu * (BL_mu1_2 % BL_mu1_1) * meas_expr_dx);

      // Assemble
      expr_assembler.clearRhs();
      expr_assembler.assemble(BL_mu1);
      expr_assembler.clearMatrix();
      expr_assembler.assemble(bilin_mu_1);
      // gsMatrix<> matrix = expr_assembler.matrix();
      // gsInfo << "First part of the matrix computed : \n" << matrix <<
      // std::endl;

      /////////////////////////////////////////
      // This section is meant for DEBUGGING //
      /////////////////////////////////////////

      if (fd_test) {
        expr_assembler.clearRhs();
        expr_assembler.assemble(BL_lambda);
        const auto lame_lambda_rhs = expr_assembler.rhs();
        gsFileData<> fddx("dx." + fn);
        gsInfo << "Loaded file " << fddx.lastPath() << "\n";
        gsMultiPatch<> mpdx;
        fddx.getId(mp_id, mpdx);
        gsBoundaryConditions<> bcdx;
        fd.getId(bc_id, bcdx);
        bc.setGeoMap(mpdx);
        auto rhs_of_fd_system = ComputeSensitivityFD(
            mpdx, Aopt, function_basis, lame_lambda, lame_mu, solVector, bc);

        // gsInfo << "FD Approximation for lambda part of matrix assembly is
        // :\n"
        //        << (rhs_of_fd_system - lame_lambda_rhs) / ddx << std::endl;
      }

      // 2. Bilinear form of mu (first part)
      auto BL_mu2_1 = ijac(solution_expression, geom_expr).cwisetr();
      auto BL_mu2_2 = ijac(u_trial, geom_expr);
      auto BL_mu2 = lame_mu * (BL_mu2_2 % BL_mu2_1) * meas_expr;

      // Linear Form Part
      auto LF_1 = -rho * u_trial * ff * meas_expr;

      // Assemble
    }
  }

  return EXIT_SUCCESS;

  //! [Here starts the work in progress ]
  ///////////////////////////////////////
  // Differentiating the linear system //
  ///////////////////////////////////////
  bool evaluate_expr_v{false};
  if (evaluate_expr_v) {
    // For local evaluation prior to testing
    gsExprEvaluator<> evaluator{expr_assembler};
    gsMatrix<> evalPoint(2, 1);
    evalPoint << .5, .5;

    // Modified / simplified expressions:
    auto d_jac_d_c = jac(u_trial);  // assumes isoparametrics
    auto inv_jac = jac(geom_expr).inv();
    auto d_jac_inv_d_c = -inv_jac * d_jac_d_c;

    // Differntiating lambda
    auto bilin_lambda_der1 =
        (d_jac_inv_d_c * ijac(solution_expression, geom_expr)).trace() *
        idiv(u_trial, geom_expr).tr();
    // auto bilin_lambda_der2 =
    //     idiv(solution_expression, geom_expr) * (d_jac_inv_d_c *
    //     igrad(u_trial, geom_expr).tr());
    auto bilin_lambda_der = lame_lambda * (bilin_lambda_der1)*meas(geom_expr);

    auto derivative = bilin_lambda_der;

    gsInfo
        << "\nFirst Expression:\n"
        << evaluator.eval(
               (d_jac_inv_d_c * ijac(solution_expression, geom_expr)).trace() *
                   idiv(u_trial, geom_expr).tr(),
               evalPoint)
        << "\n";

    return EXIT_SUCCESS;
  }
}  // end main
