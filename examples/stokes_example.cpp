/** @file stokes_example.cpp

    @brief Steady Stokes problem with adjoint approach for sensitivity
   analysis
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

  // field IDs
  constexpr index_t PRESSURE_ID = 0;
  constexpr index_t VELOCITY_ID = 1;
  // field dimensions
  constexpr index_t PRESSURE_DIM = 1;
  // number of solution and test spaces
  constexpr index_t NUM_TRIAL = 2;
  constexpr index_t NUM_TEST = 2;

  // Setup values for timing
  double setup_time(0), assembly_time_ls(0), solving_time_ls(0),
      plotting_time(0);
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
  real_t viscosity{10};
  cmd.addReal("v", "visc", "Viscosity", viscosity);

  // Mesh options
  index_t degreeElevate = 0;
  cmd.addInt("e", "degreeElevate", "Number of uniform degree elevations",
             degreeElevate);
  index_t numRefine = 0;
  cmd.addInt("r", "uniformRefine", "Number of uniform h-refinement loops",
             numRefine);

  std::string fn("../../filedata/pde/stokes2d_bvp.xml");
  cmd.addString("f", "file", "Input XML file", fn);

  // A few more mesh options
  index_t mp_id{0}, vel_bc_id{1}, ass_opt_id{10};
  cmd.addInt("m", "multipach_id", "ID of the multipatch mesh in mesh file",
             mp_id);
  cmd.addInt("b", "boundary_id",
             "ID of the boundary condition function in mesh file", vel_bc_id);
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
  gsInfo << "Loaded file " << fd.lastPath() << std::endl;
  // retrieve multi-patch data
  gsMultiPatch<> domain_patches;
  fd.getId(mp_id, domain_patches);
  // retrieve velocity boundary conditions from file
  gsBoundaryConditions<> velocity_bcs;
  fd.getId(vel_bc_id, velocity_bcs);
  velocity_bcs.setGeoMap(domain_patches);
  gsInfo << "Velocity boundary conditions:\n" << velocity_bcs << std::endl;
  // retrieve assembly options
  gsOptionList Aopt;
  fd.getId(ass_opt_id, Aopt);

  // Debugging boundary conditions
  // for (typename gsBoundaryConditions<>::const_iterator cit =
  //          velocity_bcs.dirichletBegin();
  //      cit != velocity_bcs.dirichletEnd(); cit++) {
  //   gsInfo << cit->patch() << " " << cit->side() << " " << cit->unknown() <<
  //   " "
  //          << cit->unkComponent() << " " << *(cit->function()) << std::endl;
  // }

  const index_t geomDim = domain_patches.geoDim();
  gsInfo << "Geometric dimension " << geomDim << std::endl;

  //! [Refinement]
  gsMultiBasis<> function_basis_velocity(
      domain_patches,
      true);  // true: poly-splines (not NURBS)
  gsMultiBasis<> function_basis_pressure(
      domain_patches,
      true);  // true: poly-splines (not NURBS)

  // Degree elevation
  for (int e = 0; e < degreeElevate; e++) {
    function_basis_velocity.degreeElevate();
    function_basis_pressure.degreeElevate();
  }

  // Taylor-Hood-ing that B****
  function_basis_velocity.degreeElevate();

  // h-refine each basis (for performing the analysis)
  for (int r = 0; r < numRefine; ++r) {
    function_basis_velocity.uniformRefine();
    function_basis_pressure.uniformRefine();
  }

  // Output user information
  gsInfo << "Summary Velocity :\nPatches: " << domain_patches.nPatches()
         << "\nMin-degree : " << function_basis_velocity.minCwiseDegree()
         << "\nMax-degree: " << function_basis_velocity.maxCwiseDegree() << "\n"
         << std::endl;
  gsInfo << "Summary Pressure :\nPatches: " << domain_patches.nPatches()
         << "\nMin-degree : " << function_basis_pressure.minCwiseDegree()
         << "\nMax-degree: " << function_basis_pressure.maxCwiseDegree() << "\n"
         << std::endl;
#ifdef _OPENMP
  gsInfo << "Available threads: " << omp_get_max_threads() << std::endl;
  omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
  gsInfo << "Number of threads: " << omp_get_num_threads() << std::endl;
#endif

  ///////////////////
  // Problem Setup //
  ///////////////////

  // Construct expression assembler
  // (takes number of test and solution function spaces as arguments)
  gsExprAssembler<> expr_assembler(NUM_TEST, NUM_TRIAL);
  expr_assembler.setOptions(Aopt);
  gsInfo << "Active options:\n" << expr_assembler.options() << std::endl;

  // Elements used for numerical integration
  expr_assembler.setIntegrationElements(function_basis_velocity);

  // Set the geometry map
  geometryMap geom_expr = expr_assembler.getMap(domain_patches);

  // Set the discretization space
  space p_trial = expr_assembler.getSpace(function_basis_pressure, PRESSURE_DIM,
                                          PRESSURE_ID);
  gsInfo << "Solution space for pressure (id=" << p_trial.id() << ") has "
         << p_trial.rows() << " rows and " << p_trial.cols() << " columns."
         << std::endl;
  space u_trial =
      expr_assembler.getSpace(function_basis_velocity, geomDim, VELOCITY_ID);
  gsInfo << "Solution space for velocity (id=" << u_trial.id() << ") has "
         << u_trial.rows() << " rows and " << u_trial.cols() << " columns."
         << std::endl;

  // Solution vector and solution variable
  gsMatrix<> pressure_solution;
  solution pressure_solution_expression =
      expr_assembler.getSolution(p_trial, pressure_solution);
  gsMatrix<> velocity_solution;
  solution velocity_solution_expression =
      expr_assembler.getSolution(u_trial, velocity_solution);

  // Intitalize multi-patch interfaces for pressure field
  p_trial.setup(0);
  // Initialize interfaces and Dirichlet bcs for velocity field
  u_trial.setup(velocity_bcs, Aopt.getInt("DirichletValues"), 0);

  // Initialize the system
  expr_assembler.initSystem();
  setup_time += timer.stop();

  gsInfo << "Number of degrees of freedom : " << expr_assembler.numDofs()
         << std::endl;
  gsInfo << "Number of blocks in the system matrix : "
         << expr_assembler.numBlocks() << std::endl;

  //////////////
  // Assembly //
  //////////////
  gsInfo << "Starting assembly of linear system ..." << std::flush;
  timer.restart();

  // Compute the system matrix and right-hand side
  auto phys_jacobian = ijac(u_trial, geom_expr);
  auto bilin_conti = p_trial * idiv(u_trial, geom_expr).tr() * meas(geom_expr);
  auto bilin_press = idiv(u_trial, geom_expr) * p_trial.tr() * meas(geom_expr);
  auto bilin_mu_1 = viscosity * (phys_jacobian.cwisetr() % phys_jacobian.tr()) *
                    meas(geom_expr);
  auto bilin_mu_2 =
      viscosity * (phys_jacobian % phys_jacobian.tr()) * meas(geom_expr);

  expr_assembler.assemble(bilin_conti, bilin_press, bilin_mu_1, bilin_mu_2);

  assembly_time_ls += timer.stop();
  gsInfo << "\t\tFinished" << std::endl;

  ///////////////////
  // Linear Solver //
  ///////////////////

  gsInfo << "Solving the linear system of equations ..." << std::flush;
  timer.restart();
  const auto& system_matrix = expr_assembler.matrix();
  const auto& rhs_vector = expr_assembler.rhs();

  // Initialize linear solver
  // gsSparseSolver<>::CGDiagonal solver;
  // gsSparseSolver<>::BiCGSTABILUT solver;
  gsSparseSolver<>::BiCGSTABDiagonal solver;
  solver.compute(system_matrix);
  gsMatrix<> complete_solution = solver.solve(rhs_vector);

  solving_time_ls += timer.stop();
  gsInfo << "\tFinished" << std::endl;

  pressure_solution = complete_solution;
  velocity_solution = complete_solution;

  //////////////////////////////
  // Export and Visualization //
  //////////////////////////////
  gsExprEvaluator<> expression_evaluator(expr_assembler);

  // // Old-school export
  // expression_evaluator.writeParaview(velocity_solution_expression, geom_expr,
  //                                    "velocity");
  // expression_evaluator.writeParaview(pressure_solution_expression, geom_expr,
  //                                    "pressure");

  // Generate Paraview File
  gsInfo << "Starting the paraview export ..." << std::flush;
  timer.restart();
  if (plot) {
    gsParaviewCollection collection("ParaviewOutput/solution",
                                    &expression_evaluator);
    collection.options().setSwitch("plotElements", true);
    collection.options().setInt("plotElements.resolution", sample_rate);
    collection.newTimeStep(&domain_patches);
    collection.addField(velocity_solution_expression, "velocity");
    collection.addField(pressure_solution_expression, "pressure");
    collection.saveTimeStep();
    collection.save();

  } else {
    gsInfo << "skipping";
  }
  gsInfo << "\tFinished" << std::endl;

  // Export solution file as xml
  gsInfo << "Starting the xml export ..." << std::flush;
  if (export_xml) {
    gsMatrix<> full_solution_velocity;
    gsFileData<> output;
    output << velocity_solution;  // only computed quantities without fixed BCs
    velocity_solution_expression.extractFull(
        full_solution_velocity);  // patch-wise solution with BCs
    output << full_solution_velocity;
    output.save("velocity_field.xml");
    gsMatrix<> full_solution_pressure;
    gsFileData<> output_pressure;
    output_pressure
        << pressure_solution;  // only computed quantities without fixed BCs
    pressure_solution_expression.extractFull(
        full_solution_pressure);  // patch-wise solution with BCs
    output_pressure << full_solution_pressure;
    output_pressure.save("pressure_field.xml");
  } else {
    gsInfo << "skipping";
  }
  plotting_time += timer.stop();
  gsInfo << "\t\tFinished" << std::endl;

  // User output infor timings
  gsInfo << "\n\nTotal time: "
         << setup_time + assembly_time_ls + solving_time_ls + plotting_time
         << std::endl;
  gsInfo << "                       Setup: " << setup_time << std::endl;
  gsInfo << "      Assembly Linear System: " << assembly_time_ls << std::endl;
  gsInfo << "       Solving Linear System: " << solving_time_ls << std::endl;
  gsInfo << "                    Plotting: " << plotting_time << std::endl
         << std::flush;

  return EXIT_SUCCESS;

}  // end main
