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

// Global Typedefs
typedef gsExprAssembler<>::geometryMap geometryMap;
typedef gsExprAssembler<>::variable variable;
typedef gsExprAssembler<>::space space;
typedef gsExprAssembler<>::solution solution;

int main(int argc, char* argv[]) {
  ////////////////////
  // Global Options //
  ////////////////////
  constexpr const int solution_field_dimension{2};

  // Setup values for timing
  double setup_time(0), assembly_time_ls(0), solving_time_ls(0),
      assembly_time_adj_ls(0), solving_time_adj_ls(0),
      objective_function_time(0), plotting_time(0);
  gsStopwatch timer;
  timer.restart();

  ////////////////////////////////
  // Parse Command Line Options //
  ////////////////////////////////
  // Title
  gsCmdLine cmd("Stokes Example");

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

  // Material constants
  real_t viscosity{2000000}, rho{1000};
  cmd.addReal("L", "firstLame", "First Lame constant, material parameter",
              viscosity);
  cmd.addReal("R", "rho", "Density", rho);

  // Mesh options
  index_t numRefine = 0;
  cmd.addInt("r", "uniformRefine", "Number of Uniform h-refinement loops",
             numRefine);

  std::string fn("../playground/linear_elasticity/askew_rectangle_mesh.xml");
  cmd.addString("f", "file", "Input XML file", fn);

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
  gsExprAssembler<> expr_assembler(2, 2);
  expr_assembler.setOptions(Aopt);
  gsInfo << "Active options:\n" << expr_assembler.options() << "\n";

  // Elements used for numerical integration
  expr_assembler.setIntegrationElements(function_basis);

  // Set the geometry map
  geometryMap geom_expr = expr_assembler.getMap(mp);

  // Set the discretization space
  space u_trial =
      expr_assembler.getSpace(function_basis, solution_field_dimension, 1);
  space p_trial =
      expr_assembler.getSpace(function_basis, 1, 0);  // Might be 2 as v is 2D

  // Set the source term
  auto ff = expr_assembler.getCoeff(f, geom_expr);

  // Solution vector and solution variable
  gsMatrix<> velocity_solution;
  solution velocity_solution_expression =
      expr_assembler.getSolution(u_trial, velocity_solution);
  gsMatrix<> pressure_solution;
  solution pressure_solution_expression =
      expr_assembler.getSolution(p_trial, pressure_solution);

  // Initialize Dirichlet bcs for velocity field
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

  // Debug
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

  // Compute the system matrix and right-hand side
  auto phys_jacobian = ijac(u_trial, geom_expr);
  auto bilin_conti = p_trial * idiv(u_trial, geom_expr).tr() * meas(geom_expr);
  auto bilin_press = idiv(u_trial, geom_expr) * p_trial.tr() * meas(geom_expr);
  auto bilin_mu_1 = viscosity * (phys_jacobian.cwisetr() % phys_jacobian.tr()) *
                    meas(geom_expr);
  auto bilin_mu_2 =
      viscosity * (phys_jacobian % phys_jacobian.tr()) * meas(geom_expr);

  print_function_expressions("bilin_mu_1", bilin_mu_1);
  expr_assembler.assemble(bilin_mu_2);

  std::ofstream matrix("matrix.tsv");
  matrix << static_cast<gsMatrix<>>(expr_assembler.matrix()) << std::endl;
  matrix.close();

  expr_assembler.assemble(bilin_conti, bilin_press, bilin_mu_1);

  assembly_time_ls += timer.stop();
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
  gsMatrix<> complete_solution;
  complete_solution = solver.solve(expr_assembler.rhs());

  // Daniel, please ignore this
  gsInfo << "complete_solution.size() : " << complete_solution.size();
  gsInfo << "What is freesize " << p_trial.mapper().freeSize();
  pressure_solution =
      complete_solution.block(0, 0, p_trial.mapper().freeSize(), 1);
  gsInfo << "Yeahi 1 " << std::endl;
  velocity_solution = complete_solution.block(0, p_trial.mapper().freeSize(),
                                              u_trial.mapper().freeSize(), 1);

  solving_time_ls += timer.stop();
  gsInfo << "\tFinished" << std::endl;

  //////////////////////////////
  // Export and Visualization //
  //////////////////////////////
  gsExprEvaluator<> expression_evaluator(expr_assembler);

  // Generate Paraview File
  gsInfo << "Starting the paraview export ..." << std::flush;
  timer.restart();
  if (plot) {
    gsParaviewCollection collection("ParaviewOutput/solution",
                                    &expression_evaluator);
    collection.options().setSwitch("plotElements", true);
    collection.options().setInt("plotElements.resolution", sample_rate);
    collection.newTimeStep(&mp);
    collection.addField(velocity_solution_expression, "velocity");
    collection.addField(pressure_solution_expression, "pressure");
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
    output << velocity_solution;
    velocity_solution_expression.extractFull(full_solution);
    output << full_solution;
    output.save("solution-field.xml");
  } else {
    gsInfo << "skipping";
  }
  plotting_time += timer.stop();
  gsInfo << "\t\tFinished" << std::endl;

  // User output infor timings
  gsInfo << "\n\nTotal time: "
         << setup_time + assembly_time_ls + solving_time_ls +
                assembly_time_adj_ls + solving_time_adj_ls +
                objective_function_time + plotting_time
         << "\n";
  gsInfo << "                       Setup: " << setup_time << "\n";
  gsInfo << "      Assembly Linear System: " << assembly_time_ls << "\n";
  gsInfo << "       Solving Linear System: " << solving_time_ls << "\n";
  gsInfo << "                    Plotting: " << plotting_time << "\n"
         << std::flush;

  return EXIT_SUCCESS;

}  // end main
