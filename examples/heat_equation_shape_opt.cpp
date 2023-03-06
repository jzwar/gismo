/** @file poisson2_example.cpp

    @brief Example File showing how to use expression assembler

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): A. Mantzaflaris
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
                          const gsBoundaryConditions<>& bc) {
  gsExprAssembler<> expr_assembler(1, 1);
  expr_assembler.setOptions(Aopt);

  // Elements used for numerical integration
  expr_assembler.setIntegrationElements(function_basis);

  // Set the geometry map
  geometryMap G = expr_assembler.getMap(mp_dx);

  // Set the discretization space
  space u_trial = expr_assembler.getSpace(function_basis, 1);

  u_trial.setup(bc, dirichlet::l2Projection, 0);
  // Compute the system matrix and right-hand side
  auto bilin_form = thermal_diffusivity * igrad(u_trial, G) *
                    igrad(u_trial, G).tr() * meas(G);

  // Assemble
  expr_assembler.initSystem();
  expr_assembler.assemble(bilin_form);  // matrix

  gsInfo << "Number of DOFs in the fd system : " << expr_assembler.numDofs()
         << std::endl;

  // Return the matrix to evaluate the residual
  return expr_assembler.matrix();
}

void ComputeSensitivities(const gsMultiPatch<>& mp,
                          const solution& solution_expression,
                          const gsOptionList& Aopt,
                          const gsMultiBasis<>& function_basis) {
  gsExprAssembler<> expr_assembler(1, 1);
  expr_assembler.setOptions(Aopt);

  // Elements used for numerical integration
  expr_assembler.setIntegrationElements(function_basis);

  // Set the geometry map
  geometryMap G = expr_assembler.getMap(mp);

  // Set the discretization space
  space u_trial = expr_assembler.getSpace(function_basis, 1);

  gsMatrix<> solVector{solution_expression.coefs()};
  solution solution_solved = expr_assembler.getSolution(u_trial, solVector);

  // Define expressions
  auto bilinear_form_of_solution =
      igrad(u_trial, G) * igrad(u_trial, G).tr() * meas(G);

  // Assemble
  expr_assembler.initSystem();
  expr_assembler.assemble(bilinear_form_of_solution);
}

//! [Include namespace]
int main(int argc, char* argv[]) {
  //
  constexpr const int solution_field_dimension{1};
  ////////////////////////////////
  // Parse Command Line Options //
  ////////////////////////////////

  gsCmdLine cmd("Heat Equation with to prepare for Sensitivities");

  // Provide vtk data
  bool plot = true;
  cmd.addSwitch("plot",
                "Create a ParaView visualization file with the solution", plot);
  bool export_xml = false;
  cmd.addSwitch("export-xml", "Export solution into g+smo xml format.",
                export_xml);

  // Material Constants
  real_t thermal_diffusivity{100.};  // e.g. Copper 100
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

  //! [Problem setup]
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

  //! [Problem setup]

  //! [Solver loop]
  gsSparseSolver<>::CGDiagonal solver;
  gsInfo << "(dot1=assembled, dot2=solved)\n"
            "\nDoFs: ";
  double setup_time(0), ma_time(0), slv_time(0);
  gsStopwatch timer;

  u_trial.setup(bc, dirichlet::l2Projection, 0);

  // Initialize the system
  expr_assembler.initSystem();
  setup_time += timer.stop();

  gsInfo << expr_assembler.numDofs() << "\n" << std::flush;

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
                             u_trial * g_N.val() * nv(G).norm());

  ma_time += timer.stop();

  gsInfo << "." << std::flush;  // Assemblying done

  timer.restart();
  const auto& matrix_in_initial_configuration = expr_assembler.matrix();
  const auto rhs_vector = expr_assembler.rhs();
  solver.compute(matrix_in_initial_configuration);
  solVector = solver.solve(expr_assembler.rhs());
  slv_time += timer.stop();
  gsInfo << "." << std::flush;  // Linear solving done

  // User output infor timings
  gsInfo << "\n\nTotal time: " << setup_time + ma_time + slv_time << "\n";
  gsInfo << "     Setup: " << setup_time << "\n";
  gsInfo << "  Assembly: " << ma_time << "\n";
  gsInfo << "   Solving: " << slv_time << "\n" << std::flush;

  gsInfo << "Print out the residual of the linear system:\n "
         << (matrix_in_initial_configuration * solVector - expr_assembler.rhs())
         << std::endl;

  //! [Starting the assembly of the bilinear folution]
  gsInfo << "\nThe assembled bilinear Form after solving:" << std::endl;

  if (fd_test) {
    gsFileData<> fddx("dx." + fn);
    gsInfo << "Loaded file " << fddx.lastPath() << "\n";
    gsMultiPatch<> mpdx;
    fddx.getId(mp_id, mpdx);
    gsBoundaryConditions<> bcdx;
    fd.getId(bc_id, bcdx);
    bc.setGeoMap(mpdx);
    auto matrix_of_fd_system(ComputeSensitivityFD(mpdx, Aopt, function_basis,
                                                  thermal_diffusivity, bc));

    gsInfo << "Print out the residual of the linear system:\n "
           << ((matrix_in_initial_configuration - matrix_of_fd_system) *
               solVector) *
                  1000
           << std::endl;
  }
  // //! [Starting the sensitivity analysis]

  ///////////////// DEBUGGER
  // For local evaluation prior to testing
  gsExprEvaluator<> evaluator{expr_assembler};
  gsMatrix<> evalPoint(2, 1);
  evalPoint << .25, .6;

  // Some expressions
  // Splitting the bilinear form into several parts to facilitate debugging
  // thermal_diffusivity *igrad(solution_expression, G) * igrad(u_trial,
  // G).cwisetr() *meas(G)

  // Auxiliary expressions
  space u_geom = expr_assembler.getSpace(function_basis, mp.geoDim());
  auto jacobian = jac(G);
  auto inv_jacs = jacobian.ginv();
  auto djacdc = jac(u_geom);
  auto grad_sol = grad(solution_expression);

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
  auto dBL_dx1 = thermal_diffusivity * BL2 * dBL1dC.tr() * BL3;
  auto dBL_dx3 = thermal_diffusivity * (BL2 * BL1.tr()) * dBL3dC.tr();

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
  };

  // print_function_expressions("Jacs", jacobian);
  // print_function_expressions("inv_jacs", inv_jacs);
  // print_function_expressions("djacdc", djacdc);
  // print_function_expressions("aux_expr", aux_expr);

  // print_function_expressions("grad(sol)", grad_sol);

  // print_function_expressions("BL1", BL1);
  // print_function_expressions("BL2", BL2);
  // print_function_expressions("BL3", BL3);

  // print_function_expressions("dBL1dC", dBL1dC);
  // print_function_expressions("dBL2dC", dBL2dC);
  // print_function_expressions("dBL3dC", dBL3dC);

  // print_function_expressions("dEXRPBL1dC",
  //                            thermal_diffusivity * BL2 * dBL1dC.tr() * BL3);
  // print_function_expressions("bilinear form of solution",
  //                            thermal_diffusivity * BL2 * BL1.tr() * BL3);
  // print_function_expressions(
  //     "dEXRPBL3dC", thermal_diffusivity * (BL2 * BL1.tr()) * dBL3dC.tr());

  // print_function_expressions("dEXRPBL2dC", dBL2dC % BL1.tr());

  // auto dBL_dx1 = thermal_diffusivity * BL2 * dBL1dC.tr() * BL3;
  // auto dBL_dx3 = thermal_diffusivity * (BL2 * BL1.tr()) * dBL3dC.tr();

  expr_assembler.clearRhs();
  expr_assembler.assemble(thermal_diffusivity * BL2 * BL1.tr() * BL3);
  gsInfo << "Bilinear form of solution residual \n"
         << expr_assembler.rhs() - rhs_vector << std::endl;
  expr_assembler.clearMatrix();
  expr_assembler.assemble(dBL_dx1);
  expr_assembler.assemble(dBL_dx3);
  gsInfo << "Bilinear derivative of solution \n"
         << expr_assembler.matrix() << std::endl;

  //! [Export visualization in ParaView]
  gsExprEvaluator<> expression_evaluator(expr_assembler);
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
  //! [Export visualization in ParaView]

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

  return EXIT_SUCCESS;

}  // end main
