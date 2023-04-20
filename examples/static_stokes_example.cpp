/** @file stokes_example.cpp

    @brief Steady Stokes problem with adjoint approach for sensitivity
   analysis

   Geometry and boundary conditions are hard-coded in the frontend
*/

#include <gismo.h>

using namespace gismo;

// Global Typedefs
typedef gsExprAssembler<>::geometryMap geometryMap;
typedef gsExprAssembler<>::variable variable;
typedef gsExprAssembler<>::space space;
typedef gsExprAssembler<>::solution solution;

int main(int argc, char* argv[]) {

    ///////////////
    // CONSTANTS //
    ///////////////

    // number of spatial dimensions
    constexpr index_t SPATIAL_DIM = 2;
    // field IDs
    constexpr index_t PRESSURE_ID = 0;
    constexpr index_t VELOCITY_ID = 1;
    // field dimensions
    constexpr index_t PRESSURE_DIM = 1;
    constexpr index_t VELOCITY_DIM = SPATIAL_DIM;
    // number of solution and test spaces
    constexpr index_t NUM_TRIAL = 2;
    constexpr index_t NUM_TEST = 2;
    // spline degree for representing the computational domain
    constexpr index_t SPLINE_DEGREE = 2;

    ///////////
    // SETUP //
    ///////////

    // Setup variables for timing
    real_t setup_time(0), assembly_time_ls(0), solving_time_ls(0),
        assembly_time_adj_ls(0), solving_time_adj_ls(0),
        objective_function_time(0), plotting_time(0);
    gsStopwatch timer;
    timer.restart();

    //////////////////////////
    // COMMAND LINE OPTIONS //
    //////////////////////////

    // Title
    gsCmdLine cmd("Static Stokes Example");

    // provide vtk data
    bool plot = false;
    cmd.addSwitch("plot", 
        "Create a ParaView visualization file with the solution", plot);
    // provide xml data
    bool export_xml = false;
    cmd.addSwitch("export-xml", "Export solution into g+smo xml format.",
        export_xml);
    // discretization for spline export
    index_t sample_rate{4};
    cmd.addInt("sample-rate", "Sample rate of splines for export", 
        sample_rate);
    // Material constants
    real_t viscosity{1e-6};
    cmd.addReal("v", "visc", "Viscosity", viscosity);
    // Mesh options
    index_t num_refine = 0;
    cmd.addInt("r", "uniformRefine", "Number of uniform h-refinement loops",
        num_refine);

#ifdef _OPENMP
    // make number of openmp threads configurable
    index_t n_omp_threads{1};
    cmd.addInt("p", "n_threads", "Number of threads used", n_omp_threads);
#endif

    // Parse command line options
    try {
        cmd.getValues(argc, argv);
    } catch (int rv) {
        return rv;
    }

    gsInfo << "Starting g+smo fronend " << std::quoted(cmd.getMessage()) 
           << std::endl << std::endl;

    //////////////
    // GEOMETRY //
    //////////////

    gsInfo << "Constructing geometry ..." << std::endl;
    // Represent the unit square as a multi-patch BSpline of degree 2
    gsMultiPatch<> patches(*gsNurbsCreator<>::BSplineSquareDeg(SPLINE_DEGREE));
    patches.computeTopology();
    gsInfo << "... Done." << std::endl;

    /////////////////////////
    // BOUNDARY CONDITIONS //
    /////////////////////////

    gsInfo << "Setting boundary conditions ..." << std::endl;
    // empty boundary condition container
    gsBoundaryConditions<> pressureBcInfo;
    // no boundary conditions, but required for identifying pressure continuity 
    // across patches
    pressureBcInfo.setGeoMap(patches);
    // empty boundary condition container
    gsBoundaryConditions<> velocityBcInfo;
    // function for the no-slip boundaries 
    // (first parameter is the value, second parameter the domain dimension)
    gsConstantFunction<> g_noslip(gsVector<>(0.0,0.0), SPATIAL_DIM);
    // gsConstantFunction<> g_noslip(0.0, 0.0, SPATIAL_DIM);
    // function for the slip part of the boundary
    gsConstantFunction<> g_wallslip(gsVector<>(1.0,0.0), SPATIAL_DIM);
    // gsConstantFunction<> g_wallslip(1.0, 0.0, SPATIAL_DIM);
    // assign the boundary conditions
    velocityBcInfo.addCondition(0, boundary::west,  condition_type::dirichlet, 
        &g_noslip);
    velocityBcInfo.addCondition(0, boundary::east,  condition_type::dirichlet, 
        &g_noslip);
    velocityBcInfo.addCondition(0, boundary::north, condition_type::dirichlet, 
        &g_wallslip);
    velocityBcInfo.addCondition(0, boundary::south, condition_type::dirichlet, 
        &g_noslip);
    // set the geometric map for the boundary conditons
    velocityBcInfo.setGeoMap(patches);
    gsInfo << "... Done." << std::endl;

    //////////////
    // ANALYSIS //
    //////////////

    gsInfo << "Setting up function basis for analysis ..." << std::endl;
    // Function basis for analysis
    // (boolean true indicates use of poly-splines, i.e., not NURBS)
    gsMultiBasis<> function_basis(patches, true);
    // h-refine each basis (for performing the analysis)
    for (int r = 0; r < num_refine; ++r) {
        function_basis.uniformRefine();
    }
    gsInfo << "... Done." << std::endl;

    ////////////////////////
    // PROBLEM DEFINITION //
    ////////////////////////

    gsInfo << "Initializing the problem ..." << std::endl;
    // Construct expression assembler 
    // (takes number of test and solution function spaces as arguments)
    gsExprAssembler<> expr_assembler(NUM_TEST, NUM_TRIAL);
    // Use (refined) function basis for numerical integration
    expr_assembler.setIntegrationElements(function_basis);
    // Perform the computations on the multi-patch geometry
    geometryMap geom_expr = expr_assembler.getMap(patches);
    // Define the discretization of the unknown fields
    space p_trial =
        expr_assembler.getSpace(function_basis, PRESSURE_DIM, PRESSURE_ID);
    space u_trial =
        expr_assembler.getSpace(function_basis, VELOCITY_DIM, VELOCITY_ID);
    // Set the boundary conditions for the pressure field
    p_trial.setup(pressureBcInfo, dirichlet::l2Projection);
    // Set the boundary conditions for the velocity field
    u_trial.setup(velocityBcInfo, dirichlet::l2Projection);
    // Initialize the system
    expr_assembler.initSystem();
    setup_time += timer.stop();
    gsInfo << "... Done." << std::endl;

    ///////////////
    // USER INFO //
    ///////////////

    gsInfo << std::endl << "Problem summary for " 
           << std::quoted(cmd.getMessage()) << std::endl;
    gsInfo << "- Geometry" << std::endl
           << "\t- Number of dimensions: " << patches.geoDim() << std::endl
           << "\t- Number of patches: " << patches.nPatches() << std::endl 
           << "\t\t- Minimum degree: " << function_basis.minCwiseDegree() 
           << std::endl 
           << "\t\t- Maximum degree: " << function_basis.maxCwiseDegree() 
           << std::endl;
    gsInfo << "- Problem" << std::endl
        //    << "\t- Active assembly options: " << expr_assembler.options() 
        //    << std::endl
           << "\t- Total number of unknown fields (corresponding to blocks in the system matrix): " 
           << expr_assembler.numBlocks() << std::endl
           << "\t\t- Solution space for pressure " << std::endl
           << "\t\t\t- Field ID: " << p_trial.id() << std::endl
           << "\t\t\t- Field dimensions: (" << p_trial.rows() << " x " 
           << p_trial.cols() << ")" << std::endl
           << "\t\t- Solution space for velocity " << std::endl
           << "\t\t\t- Field ID: " << u_trial.id() << std::endl 
           << "\t\t\t- Field dimensions: (" << u_trial.rows() << " x " 
           << u_trial.cols() << ")" << std::endl
           << "\t-Total number of degrees of freedom: " 
           << expr_assembler.numDofs() << std::endl;
#ifdef _OPENMP
  gsInfo << "\t- Available threads: " << omp_get_max_threads() << std::endl;
  omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
  gsInfo << "\t- Number of threads: " << omp_get_num_threads() << std::endl;
#endif
    gsInfo << std::endl;

    //////////////
    // ASSEMBLY //
    //////////////

    gsInfo << "Starting assembly of linear system ..." << std::endl;
    timer.restart();

    // Define all terms that contribute to the system matrix (right-hand side, 
    // e.g., source terms, are zero for this example)
    auto phys_jacobian = ijac(u_trial, geom_expr);
    auto bilin_conti = 
        p_trial * idiv(u_trial, geom_expr).tr() * meas(geom_expr);
    auto bilin_press = 
        idiv(u_trial, geom_expr) * p_trial.tr() * meas(geom_expr);
    auto bilin_mu_1 = 
        viscosity * (phys_jacobian.cwisetr() % phys_jacobian.tr()) 
        * meas(geom_expr);
    auto bilin_mu_2 =
        viscosity * (phys_jacobian % phys_jacobian.tr()) * meas(geom_expr);
    // Perform the assembly
    expr_assembler.assemble(bilin_conti, bilin_press, bilin_mu_1, bilin_mu_2);
    assembly_time_ls += timer.stop();
    gsInfo << "... Done." << std::endl;

    ///////////////////
    // LINEAR SOLVER //
    ///////////////////

    gsInfo << "Solving the linear system of equations ..." << std::endl;
    timer.restart();

    const auto& system_matrix = expr_assembler.matrix();
    const auto& rhs_vector = expr_assembler.rhs();
    // Initialize linear solver
    gsSparseSolver<>::CGDiagonal solver;
    solver.compute(system_matrix);
    // Solve the linear equation system
    gsMatrix<> complete_solution = solver.solve(rhs_vector);
    solving_time_ls += timer.stop();
    gsInfo << "... Done." << std::endl;

    // Extract the solution variables from the full solution
    // TODO: This needs to be done appropriately!!!
    gsMatrix<> pressure_solution = 
        complete_solution.block(0, 0, p_trial.mapper().freeSize(), 1);
    gsMatrix<> velocity_solution = 
        complete_solution.block(p_trial.mapper().freeSize(), 0, 
                                u_trial.mapper().freeSize(), 1);
    // print the solution matrices after extraction:
    // gsInfo << "after extraction:" << std::endl;
    // gsInfo << "pressure_solution:\n" << pressure_solution << std::endl;
    // gsInfo << "velocity_solution:\n" << velocity_solution << std::endl;

    ///////////////////
    // VISUALIZATION //
    ///////////////////

    // Solution fields for evaluation
    solution pressure_solution_expression =
        expr_assembler.getSolution(p_trial, pressure_solution);
    solution velocity_solution_expression =
        expr_assembler.getSolution(u_trial, velocity_solution);

    if (plot) {
        timer.restart();
         // Generate Paraview File
        gsInfo << "Starting the paraview export ..." << std::endl;
        // Instance for evaluating the assembled expressions
        gsExprEvaluator<> expression_evaluator(expr_assembler);
        gsParaviewCollection collection("ParaviewOutput/solution",
                                        &expression_evaluator);
        collection.options().setSwitch("plotElements", true);
        collection.options().setInt("plotElements.resolution", sample_rate);
        collection.newTimeStep(&patches);
        collection.addField(velocity_solution_expression, "velocity");
        collection.addField(pressure_solution_expression, "pressure");
        collection.saveTimeStep();
        collection.save();
        plotting_time += timer.stop();
        gsInfo << "... Done." << std::endl;
    }

    ////////////
    // EXPORT //
    ////////////

    if (export_xml) {
        // Export solution file as xml
        gsInfo << "Starting the xml export ..." << std::flush;
        gsMultiPatch<> mpsol;
        gsMatrix<> full_solution;
        gsFileData<> output;
        output << velocity_solution;
        velocity_solution_expression.extractFull(full_solution);
        output << full_solution;
        output.save("velocity-field.xml");
        gsInfo << "... Done." << std::endl;
    }

    ////////////
    // TIMING //
    ////////////

    // User output for timings
    gsInfo << "\n\nTotal time: "
            << setup_time + assembly_time_ls + solving_time_ls +
                assembly_time_adj_ls + solving_time_adj_ls +
                objective_function_time + plotting_time
            << std::endl;
    gsInfo << "                       Setup: " << setup_time << std::endl;
    gsInfo << "      Assembly Linear System: " << assembly_time_ls << std::endl;
    gsInfo << "       Solving Linear System: " << solving_time_ls << std::endl;
    gsInfo << "                    Plotting: " << plotting_time << std::endl
            << std::flush;

    return EXIT_SUCCESS;
}