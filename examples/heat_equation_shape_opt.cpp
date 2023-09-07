/** @file heat_equation_shape_opt.cpp

    @brief Laplace equation and adjoints to determine sensitivities with respect
   to the control point position and an underlying parametrization.

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

auto ComputeSensitivityFD(const gsMultiPatch<>& mp_dx, const gsOptionList& Aopt,
                          const gsMultiBasis<>& function_basis,
                          const double& thermal_diffusivity,
                          const gsMatrix<>& solVector_reference,
                          const gsBoundaryConditions<>& bc) {
  gsExprAssembler<> expr_assembler(1, 1);
  expr_assembler.setOptions(Aopt);

  // Elements used for numerical integration
  expr_assembler.setIntegrationElements(function_basis);

  // Set the geometry map
  geometryMap G = expr_assembler.getMap(mp_dx);

  // Set the discretization space
  space u_trial = expr_assembler.getSpace(function_basis, 1);

  // Solution space

  gsMatrix<> solVector{solVector_reference};
  solution solution_expression = expr_assembler.getSolution(u_trial, solVector);

  u_trial.setup(bc, dirichlet::l2Projection, 0);
  // Compute the system matrix and right-hand side

  // Assemble
  auto BL1 = igrad(solution_expression, G);
  auto BL2 = igrad(u_trial, G);
  auto BL3 = meas(G);
  expr_assembler.initSystem();
  expr_assembler.clearRhs();
  expr_assembler.assemble(thermal_diffusivity * BL2 * BL1.tr() * BL3);

  // Return the matrix to evaluate the residual
  return expr_assembler.rhs();
}

//! [Include namespace]
int main(int argc, char* argv[]) {
  ////////////////////
  // Global Options //
  ////////////////////
  constexpr const int solution_field_dimension{1};

  ////////////////////////////////
  // Parse Command Line Options //
  ////////////////////////////////
  // Title
  gsCmdLine cmd("Heat Equation with to prepare for Sensitivities");

  // Provide vtk data
  bool plot = false;
  cmd.addSwitch("plot",
                "Create a ParaView visualization file with the solution", plot);
  bool export_xml = false;
  cmd.addSwitch("export-xml", "Export solution into g+smo xml format.",
                export_xml);

  // Material Constants
  real_t thermal_diffusivity{1.};  // e.g. Copper 100
  cmd.addReal("A", "thermal_diffusivity",
              "Thermal diffusivity of the material (isotropic)",
              thermal_diffusivity);

  // Mesh options
  index_t numRefine = 0;
  cmd.addInt("r", "uniformRefine", "Number of Uniform h-refinement loops",
             numRefine);
  index_t numElevate = 0;
  cmd.addInt("e", "degreeElevation",
             "Number of degree elevation steps to perform before solving (0: "
             "equalize degree in all directions)",
             numElevate);

  std::string fn(
      "/home/zwar/Git/forks/gismo/filedata/pde/rectangle_multipatch_bcs.xml");
  cmd.addString("f", "file", "Input XML file", fn);

  // Testing
  bool fd_test{false};
  cmd.addSwitch("fd-test", "Calculate the fd solution of bilinear form",
                fd_test);

  // Problem setup
  int target_function_id{4};
  cmd.addInt("t", "target-function", "ID of the target functionin mesh file",
             target_function_id);
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

  // Objective function
  gsFunctionExpr<> target_function;
  if (compute_objective_function) {
    fd.getId(target_function_id, target_function);
  }

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
  geometryMap G = expr_assembler.getMap(mp);

  // Set the discretization space
  space u_trial =
      expr_assembler.getSpace(function_basis, solution_field_dimension);

  // Set the source term
  auto ff = expr_assembler.getCoeff(f, G);

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
  auto bilin_form = thermal_diffusivity * igrad(u_trial, G) *
                    igrad(u_trial, G).tr() * meas(G);
  auto lin_form = u_trial * ff * meas(G);

  expr_assembler.assemble(bilin_form,  // matrix
                          lin_form     // rhs vector
  );

  // Compute the Neumann terms defined on physical space
  auto g_N = expr_assembler.getBdrFunction(G);
  expr_assembler.assembleBdr(bc.get("Neumann"),
                             u_trial *         // test_function
                                 g_N.val() *   // boundary condition function
                                 nv(G).norm()  // Measure of the surface
  );
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
    expression_evaluator.writeParaview(solution_expression, G, "solution");

    gsFileManager::open("solution.pvd");
  } else {
    gsInfo << "Done. No output created, re-run with --plot to get a "
              "ParaView "
              "file containing the solution.\n";
  }

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

  ////////////////////////////////
  // Compute Objective function //
  ////////////////////////////////
  if (compute_objective_function) {
    // Using boundary integrator
    auto target_function_expression =
        expr_assembler.getCoeff(target_function, G);
    expr_assembler.clearRhs();
    // Assemble with test function and using the sum over all integrals (using
    // partition of unity), this is a bit inefficient but only on a subdomain
    expr_assembler.assembleBdr(
        bc.get("Neumann"),
        u_trial * (solution_expression - target_function_expression) *
            (solution_expression - target_function_expression) * nv(G).norm());
    const auto objective_function_value = expr_assembler.rhs().sum();
    gsInfo << "The objective function evaluates to : "
           << objective_function_value << std::endl;

    // Assemble derivatives of objective function with respect to field
    if (compute_sensitivities) {
      expr_assembler.clearRhs();
      expr_assembler.assembleBdr(
          bc.get("Neumann"),
          2 * u_trial * (solution_expression - target_function_expression) *
              nv(G).norm());
      const auto objective_function_derivative = expr_assembler.rhs();
      gsInfo << "The objective function derivative evaluates to : \n"
             << objective_function_derivative << std::endl;
      /////////////////////////////////////////////////
      // Compute sensitivities to Objective function //
      /////////////////////////////////////////////////

      // Splitting the bilinear form into several parts to facilitate debugging
      // thermal_diffusivity *igrad(solution_expression, G) * igrad(u_trial,
      // G).cwisetr() *meas(G)

      // Auxiliary expressions
      space u_geom = expr_assembler.getSpace(function_basis, mp.geoDim());
      auto jacobian = jac(G);
      auto inv_jacs = jacobian.ginv();
      auto djacdc = jac(u_geom);

      // Components
      auto BL1 = igrad(solution_expression, G);
      auto BL2 = igrad(u_trial, G);
      auto BL3 = meas(G);

      // Derivatives
      auto aux_expr = djacdc * inv_jacs;       // Correct don't touch
      auto dBL1dC = -(BL1 * aux_expr);         // Correct don't touch
      auto dBL2dC = -(BL2 * aux_expr);         // Fucking doesn't work
      auto dBL3dC = BL3 * (aux_expr).trace();  // Correct don't touch

      // Combined Derivatives as matrix
      auto dBL_dx1 =
          thermal_diffusivity * BL2 * dBL1dC.tr() * BL3;  // works as expected
      auto dBL_dx2 = thermal_diffusivity * dBL2dC * BL1.tr() *
                     BL3;  // Multiplication error
      auto dBL_dx3 = thermal_diffusivity * (BL2 * BL1.tr()) *
                     dBL3dC.tr();  // works as expected

      // Assemble individual matrices and print into standard out
      expr_assembler.initSystem();
      expr_assembler.assemble(dBL_dx1);
      gsInfo << "Bilinear derivative of solution part 1 \n"
             << expr_assembler.matrix() << std::endl;
      // expr_assembler.assemble(dBL_dx2);

      expr_assembler.clearMatrix();
      expr_assembler.assemble(dBL_dx3);
      gsInfo << "Bilinear derivative of solution part 2\n"
             << expr_assembler.matrix() << std::endl;

      expr_assembler.clearMatrix();
      expr_assembler.assemble(dBL_dx1);
      expr_assembler.assemble(dBL_dx3);
      gsInfo << "Bilinear derivative of solution - combined\n"
             << expr_assembler.matrix() << std::endl;

      /////////////////////////////////////////
      // This section is meant for DEBUGGING //
      /////////////////////////////////////////

      if (fd_test) {
        gsFileData<> fddx("dx." + fn);
        gsInfo << "Loaded file " << fddx.lastPath() << "\n";
        gsMultiPatch<> mpdx;
        fddx.getId(mp_id, mpdx);
        gsBoundaryConditions<> bcdx;
        fd.getId(bc_id, bcdx);
        bc.setGeoMap(mpdx);
        auto rhs_of_fd_system = ComputeSensitivityFD(
            mpdx, Aopt, function_basis, thermal_diffusivity, solVector, bc);

        gsInfo << "FD Approximation :\n "
               << (rhs_of_fd_system - rhs_vector) / 0.0001 << std::endl;
      }
      return 0;

      // For local evaluation prior to testing
      gsExprEvaluator<> evaluator{expr_assembler};
      gsMatrix<> evalPoint(2, 1);
      evalPoint << .25, .6;

      // Print out lambda function
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

      // Check solution
      expr_assembler.assemble(thermal_diffusivity * BL2 * BL1.tr() * BL3);
      gsInfo << "Bilinear form of solution residual \n"
             << expr_assembler.rhs() - rhs_vector << std::endl;

      print_function_expressions("Jacs", jacobian);
      print_function_expressions("inv_jacs", inv_jacs);
      print_function_expressions("aux_expr", aux_expr);

      print_function_expressions("BL1", BL1);
      print_function_expressions("BL2", BL2);
      print_function_expressions("BL3", BL3);

      print_function_expressions("dBL2dC.T", dBL2dC.tr());
      print_function_expressions("dEXRPBL2dC", dBL2dC * BL1.tr());
    }
  }

  return EXIT_SUCCESS;

}  // end main
