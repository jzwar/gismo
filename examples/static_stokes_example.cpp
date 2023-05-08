/** @file stokes_example.cpp

    @brief Steady Stokes cavity problem with an analytical solution.

    This example has been taken from the book
        Jean Donea and Antonio Huerta, 
        Finite Element Methods for Flow Problems, 
        1st ed. (John Wiley & Sons, Ltd, 2003), 
        https://doi.org/10.1002/0470013826.
    Geometry and boundary conditions are hard-coded in the frontend.
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

    // Number of spatial dimensions
    constexpr index_t SPATIAL_DIM = 2;
    // Field IDs
    constexpr index_t PRESSURE_ID = 0;
    constexpr index_t VELOCITY_ID = 1;
    // Field dimensions
    constexpr index_t PRESSURE_DIM = 1;
    constexpr index_t VELOCITY_DIM = SPATIAL_DIM;
    // Number of solution and test spaces
    constexpr index_t NUM_TRIAL = 2;
    constexpr index_t NUM_TEST = 2;
    // Spline degree for representing the computational domain
    constexpr index_t SPLINE_DEGREE = 2;
    // Constant viscosity for the analytical solution
    constexpr real_t VISCOSITY = 1.0;

    ///////////
    // SETUP //
    ///////////

    // Setup variables for timing
    real_t setup_time(0), assembly_time_ls(0), solving_time_ls(0), 
        plotting_time(0), postprocessing_time(0);
    // Create a timer instance
    gsStopwatch timer;
    timer.restart();

    //////////////////////////
    // COMMAND LINE OPTIONS //
    //////////////////////////

    // Title
    gsCmdLine cmd("Stokes Example with Analytical Solution");

    // Provide solution as vtk data
    bool plot = false;
    cmd.addSwitch("plot", "Create a ParaView visualization file with the "
                  "solution", plot);
    // Provide solution as xml data
    bool export_xml = false;
    cmd.addSwitch("export-xml", "Export solution into g+smo xml format.",
                  export_xml);
    // Discretization for spline export
    index_t sample_rate = 4;
    cmd.addInt("sample-rate", "Sample rate of splines for export", 
               sample_rate);
    // Number of degree elevations 
    // (default is quadratic pressure and cubic velocity basis)
    index_t numElevate = 0;
    cmd.addInt("e", "numElevate", "Number of uniform degree elevations",
               numElevate);
    // Number of uniform refinement steps for the convergence study
    index_t numRefine = 0;
    cmd.addInt("r", "uniformRefine", "Number of uniform h-refinement loops",
               numRefine);
    // Switch to solve the problem only once for a fixed h-refinement
    bool last = false;
    cmd.addSwitch("last", "Solve solely for the last level of h-refinement", 
                  last);
    // Export 
    std::string errorFileName("error.csv");
    cmd.addString("f", "errorFile", "Output csv file containing the computed " 
                  "L2 errors for all refinement levels", errorFileName);

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

    gsInfo << "Starting G+Smo fronend " << std::quoted(cmd.getMessage()) 
           << std::endl << std::endl;

    //////////////
    // GEOMETRY //
    //////////////

    gsInfo << "Constructing geometry ... " << std::flush;

    // Represent the unit square as a multi-patch BSpline of degree 2
    gsMultiPatch<> patches(*gsNurbsCreator<>::BSplineSquareDeg(SPLINE_DEGREE));
    patches.computeTopology();

    gsInfo << "Done." << std::endl;

    /////////////////////////
    // BOUNDARY CONDITIONS //
    /////////////////////////

    gsInfo << "Setting boundary conditions ... " << std::flush;

    // function for boundaries with zero value
    // (first parameter is the value, second parameter the domain dimension)
    gsConstantFunction<> g_zero(0.0, SPATIAL_DIM);

    // empty boundary condition container
    gsBoundaryConditions<> velocityBCs;
    // set the geometric map for the boundary conditons
    velocityBCs.setGeoMap(patches);
    // assign the boundary conditions for each boundary segment and each 
    // unknown / component
    velocityBCs.addCondition(0, boundary::west,  condition_type::dirichlet, 
        &g_zero, VELOCITY_ID, false, 0);
    velocityBCs.addCondition(0, boundary::west,  condition_type::dirichlet, 
        &g_zero, VELOCITY_ID, false, 1);
    velocityBCs.addCondition(0, boundary::east,  condition_type::dirichlet, 
        &g_zero, VELOCITY_ID, false, 0);
    velocityBCs.addCondition(0, boundary::east,  condition_type::dirichlet, 
        &g_zero, VELOCITY_ID, false, 1);
    velocityBCs.addCondition(0, boundary::south, condition_type::dirichlet, 
        &g_zero, VELOCITY_ID, false, 0);
    velocityBCs.addCondition(0, boundary::south, condition_type::dirichlet, 
        &g_zero, VELOCITY_ID, false, 1);
    velocityBCs.addCondition(0, boundary::north, condition_type::dirichlet, 
        &g_zero, VELOCITY_ID, false, 0);
    velocityBCs.addCondition(0, boundary::north, condition_type::dirichlet, 
        &g_zero, VELOCITY_ID, false, 1);
    
    // Try to fix the pressure in the upper right corner to zero
    gsBoundaryConditions<> pressureBCs;
    pressureBCs.setGeoMap(patches);
    pressureBCs.addCondition(0, boundary::west, condition_type::dirichlet,
        &g_zero, PRESSURE_ID, false, 0);
    
    gsInfo << "Done." << std::endl;

    /////////////////
    // SOURCE TERM //
    /////////////////

    // Define the constant source term for this test case
    gsFunctionExpr<> source_expr(
        "(12-24*y)*x^4+(-24+48*y)*x^3+(-48*y+72*y^2-48*y^3+12)*x^2+(-2+24*y-72*y^2+48*y^3)*x+1-4*y+12*y^2-8*y^3",
        "(8-48*y+48*y^2)*x^3+(-12+72*y-72*y^2)*x^2+(4-24*y+48*y^2-48*y^3+24*y^4)*x-12*y^2+24*y^3-12*y^4",
        SPATIAL_DIM
    );

    //////////////
    // ANALYSIS //
    //////////////

    gsInfo << "Setting up function basis for analysis ... " << std::flush;

    // Function basis for analysis (true: poly-splines (not NURBS))
    gsMultiBasis<> function_basis_pressure(patches, true);
    gsMultiBasis<> function_basis_velocity(patches, true);

    // Elevate the degree as desired by the user and increase the degree one 
    // additional time for the velocity to obtain Taylor-Hood elements
    function_basis_pressure.setDegree( 
        function_basis_pressure.maxCwiseDegree() + numElevate
    );
    function_basis_velocity.setDegree( 
        function_basis_velocity.maxCwiseDegree() + numElevate + 1
    );
    
    // h-refine each basis (for performing the analysis)
    if(last) {
        for (int r=0; r<numRefine; ++r) {
            function_basis_pressure.uniformRefine();
            function_basis_velocity.uniformRefine();
        }
    }

    gsInfo << "Done." << std::endl;

    ////////////////////////
    // PROBLEM DEFINITION //
    ////////////////////////

    gsInfo << "Defining the problem ... " << std::flush;

    // Construct expression assembler 
    // (takes number of test and solution function spaces as arguments)
    gsExprAssembler<> assembler(NUM_TEST, NUM_TRIAL);
    // Use (refined) function basis for numerical integration
    assembler.setIntegrationElements(function_basis_velocity);

    // Perform the computations on the multi-patch geometry
    geometryMap geoMap = assembler.getMap(patches);

    // Define the discretization of the unknown fields
    space trial_space_pressure = assembler.getSpace(
        function_basis_pressure, PRESSURE_DIM, PRESSURE_ID
    );
    space trial_space_velocity = assembler.getSpace(
        function_basis_velocity, VELOCITY_DIM, VELOCITY_ID
    );

    // Create the solution vector for the full problem
    gsMatrix<> complete_solution;
    // For evaluating the numerical solution fields later on, provide the trial
    // spaces as well as a reference to the solution vector
    solution numerical_pressure =
        assembler.getSolution(trial_space_pressure, complete_solution);
    solution numerical_velocity =
        assembler.getSolution(trial_space_velocity, complete_solution);
    
    // Retrieve the coefficients for the artificial source term
    auto source_term = assembler.getCoeff(source_expr, geoMap);

    // Evaluator instance to evaluate the analytical solution on the geometry
    gsExprEvaluator<> evaluator(assembler);
    // Analytical solution expressions for pressure and velocity 
    gsFunctionExpr<> analytical_pressure_expression("x*(1-x)", SPATIAL_DIM);
    gsFunctionExpr<> analytical_velocity_expression(
        " x^2*(1-x)^2*(2*y-6*y^2+4*y^3)", 
        "-y^2*(1-y)^2*(2*x-6*x^2+4*x^3)", 
        SPATIAL_DIM
    );
    // Evaluate the analytical solution expressions on the geometry
    auto analytical_pressure = 
        evaluator.getVariable(analytical_pressure_expression, geoMap);
    auto analytical_velocity = 
        evaluator.getVariable(analytical_velocity_expression, geoMap);
    
    gsInfo << "Done." << std::endl;

    ///////////////
    // USER INFO //
    ///////////////

    gsInfo << std::endl << "Problem summary for " 
           << std::quoted(cmd.getMessage()) << std::endl;
    gsInfo << "- Geometry" << std::endl
           << "\t- Number of dimensions: " << patches.geoDim() << std::endl
           << "\t- Number of patches: " << patches.nPatches() << std::endl 
           << "\t\t- Minimum degree: " 
           << function_basis_velocity.minCwiseDegree() << std::endl 
           << "\t\t- Maximum degree: " 
           << function_basis_velocity.maxCwiseDegree() << std::endl;
    gsInfo << "- Problem" << std::endl
           << "\t- Total number of unknown fields "
           << "(corresponding to blocks in the system matrix): " 
           << assembler.numBlocks() << std::endl
           << "\t\t- Solution space for pressure " << std::endl
           << "\t\t\t- Field ID: " << trial_space_pressure.id() << std::endl
           << "\t\t\t- Field dimensions: (" << trial_space_pressure.rows() 
           << " x " << trial_space_pressure.cols() << ")" << std::endl
           << "\t\t\t- Minimum degree: " 
           << function_basis_pressure.minCwiseDegree() << std::endl
           << "\t\t\t- Maximum degree: " 
           << function_basis_pressure.maxCwiseDegree() << std::endl
           << "\t\t- Solution space for velocity " << std::endl
           << "\t\t\t- Field ID: " << trial_space_velocity.id() << std::endl 
           << "\t\t\t- Field dimensions: (" << trial_space_velocity.rows() 
           << " x " << trial_space_velocity.cols() << ")" << std::endl
           << "\t\t\t- Minimum degree: " 
           << function_basis_velocity.minCwiseDegree() << std::endl
           << "\t\t\t- Maximum degree: " 
           << function_basis_velocity.maxCwiseDegree() << std::endl;
#ifdef _OPENMP
  gsInfo << "\t- Available threads: " << omp_get_max_threads() << std::endl;
  omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
  gsInfo << "\t- Number of threads: " << omp_get_num_threads() << std::endl;
#endif
    gsInfo << std::endl;

    //////////////
    // ANALYSIS //
    //////////////

    // Create a vector for storing the L2-errors, the number of dofs, and the
    // characteristic element size for all refinement levels
    gsMatrix<> l2err(numRefine+1,2), hmax(numRefine+1,2);
    gsVector<> dofs(numRefine+1);

    // Instance for writing the data to file
    std::ofstream errorFileOutput(errorFileName);
    // Write header of file
    errorFileOutput << "refinement level;" << "DoFs;"
                    << "hmax pressure;" << "l2 error pressure;"
                    << "hmax velocity;" << "l2 error velocity;"
                    << std::endl;

    // Refine the basis with every loop, solve the system and compute the error
    for (int r=0; r<=numRefine; ++r)
    {
        // Retrieve measures for the maximum cell lengths 
        // (yet only in parameter and not in physical space)
        hmax(r,PRESSURE_ID) = 
            function_basis_pressure.basis(0).getMaxCellLength();
        hmax(r,VELOCITY_ID) = 
            function_basis_velocity.basis(0).getMaxCellLength();

        ////////////////////
        // INITIALIZATION //
        ////////////////////

        gsInfo << "Initializing the problem ... " << std::flush;
        timer.restart();

        // Intitalize the multi-patch interfaces for the pressure space
        trial_space_pressure.setup(pressureBCs, dirichlet::l2Projection, 0);
        // Set the boundary conditions for the velocity field
        trial_space_velocity.setup(velocityBCs, dirichlet::l2Projection, 0);
        // Initialize the system
        assembler.initSystem();
        // retrieve the number of DoFs
        dofs[r] = assembler.numDofs();

        setup_time += timer.stop();
        gsInfo << "Done." << std::endl;

        //////////////
        // ASSEMBLY //
        //////////////

        gsInfo << "Starting assembly of linear system (#refinements=" << r 
               << ", #DoFs=" << dofs[r] << ") " << std::flush;
        timer.restart();

        // Define all terms that contribute to the system matrix
        auto phys_jacobian = ijac(trial_space_velocity, geoMap);
        // Assemble bilinear form resulting from continuity equation
        assembler.assemble(
            trial_space_pressure * idiv(trial_space_velocity, geoMap).tr()
            * meas(geoMap)
        );
        gsInfo << "." << std::flush;
        // Assemble bilinear from resulting from pressure gradient
        assembler.assemble(
            - idiv(trial_space_velocity, geoMap) * trial_space_pressure.tr()
            * meas(geoMap)
        );
        gsInfo << "." << std::flush;
        // Assemble bilinear form resulting from velocity gradient (viscous part)
        assembler.assemble(
            VISCOSITY * (phys_jacobian.cwisetr() % phys_jacobian.tr()) * 
            meas(geoMap)
        );  
        gsInfo << "." << std::flush;
        // Assemble bilinear form resulting from transposed velocity gradient 
        // (viscous part)
        assembler.assemble(
            VISCOSITY * (phys_jacobian % phys_jacobian.tr()) * meas(geoMap)
        );
        gsInfo << ". " << std::flush;
        // Assemble linear form resulting from the source term
        assembler.assemble(
            trial_space_velocity * source_term * meas(geoMap)
        );
        
        assembly_time_ls += timer.stop();
        gsInfo << "Done." << std::endl;

        ///////////////////
        // LINEAR SOLVER //
        ///////////////////

        gsInfo << "Solving the linear system of equations ... " << std::flush;
        timer.restart();

        const auto& system_matrix = assembler.matrix();
        const auto& rhs_vector = assembler.rhs();

        // Initialize linear solver
        gsSparseSolver<>::BiCGSTABDiagonal solver;
        solver.compute(system_matrix);

        // Solve the linear equation system
        complete_solution = solver.solve(rhs_vector);
        
        solving_time_ls += timer.stop();
        gsInfo << "Done." << std::endl;

        ////////////////////
        // POSTPROCESSING //
        ////////////////////

        gsInfo << "Performing postprocessing ... " << std::flush;
        timer.restart();

        // Compute the L2 error for pressure and velocity fields
        l2err(r,PRESSURE_ID) = math::sqrt( 
            evaluator.integral( 
                (analytical_pressure-numerical_pressure).sqNorm()*meas(geoMap) 
            ) 
        );
        l2err(r,VELOCITY_ID) = math::sqrt( 
            evaluator.integral( 
                (analytical_velocity-numerical_velocity).sqNorm()*meas(geoMap) 
            ) 
        );

        postprocessing_time += timer.stop();
        gsInfo << "Done." << std::endl;

        gsInfo << "L2 error (pressure): " << std::scientific 
            << std::setprecision(3) << l2err(r,PRESSURE_ID) << std::endl;
        gsInfo << "L2 error (velocity): " << std::scientific 
            << std::setprecision(3) << l2err(r,VELOCITY_ID) << std::endl 
            << std::endl;
        
        // Write to file
        errorFileOutput << r                    << ";"
                        << dofs[r]              << ";"
                        << hmax(r,PRESSURE_ID)  << ";"
                        << l2err(r,PRESSURE_ID) << ";"
                        << hmax(r,VELOCITY_ID)  << ";"
                        << l2err(r,VELOCITY_ID) << ";"
                        << std::endl;

        // Refine the bases, unless it's the last iteration
        if(r!=numRefine) {
            function_basis_pressure.uniformRefine();
            function_basis_velocity.uniformRefine();
        }
    }
    errorFileOutput.close();

    ///////////////////
    // VISUALIZATION //
    ///////////////////

    // Only the last level of refinement will be exported
    if (plot) {
        // Generate Paraview File
        gsInfo << "Starting the paraview export ... " << std::flush;
        timer.restart();

        // Instance for evaluating the assembled expressions
        gsParaviewCollection collection(
            "ParaviewOutput/static_stokes_example", &evaluator
        );
        collection.options().setSwitch("plotElements", true);
        collection.options().setInt("plotElements.resolution", sample_rate);
        collection.newTimeStep(&patches);
        collection.addField(analytical_pressure, "pressure (analytical)");
        collection.addField(analytical_velocity, "velocity (analytical)");
        collection.addField(numerical_pressure, "pressure (numerical)");
        collection.addField(numerical_velocity, "velocity (numerical)");
        collection.saveTimeStep();
        collection.save();

        plotting_time += timer.stop();
        gsInfo << "Done." << std::endl;
    }

    ////////////
    // EXPORT //
    ////////////

    if (export_xml) {
        // Export solution file as xml
        gsInfo << "Starting the xml export ... " << std::flush;

        // Export pressure
        gsMatrix<> full_solution_pressure;
        gsFileData<> pressure_data_file;
        numerical_pressure.extractFull(full_solution_pressure);
        pressure_data_file << full_solution_pressure;
        pressure_data_file.save("pressure-field.xml");

        // Export velocity
        gsMatrix<> full_solution_velocity;
        gsFileData<> velocity_data_file;
        numerical_velocity.extractFull(full_solution_velocity);
        velocity_data_file << full_solution_velocity;
        velocity_data_file.save("velocity-field.xml");
        
        gsInfo << "Done." << std::endl;
    }

    ////////////
    // TIMING //
    ////////////

    // User output for timings
    gsInfo << "\n\nTotal time: "
           << setup_time + assembly_time_ls + solving_time_ls +
              postprocessing_time + plotting_time
           << std::endl;
    gsInfo << "                       Setup: " << setup_time << std::endl;
    gsInfo << "      Assembly Linear System: " << assembly_time_ls << std::endl;
    gsInfo << "       Solving Linear System: " << solving_time_ls << std::endl;
    gsInfo << "             Post-Processing: " << postprocessing_time << std::endl;
    gsInfo << "                    Plotting: " << plotting_time << std::endl
           << std::flush;

    return EXIT_SUCCESS;
}