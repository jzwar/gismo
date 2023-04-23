/** @file double_poisson_example.cpp

    @brief Test case solving two independent Poisson problems monolithically to
    emulate the solution of multi-field problems in G+smo.
*/

#include <gismo.h>

using namespace gismo;
typedef gsExprAssembler<>::geometryMap geometryMap;
typedef gsExprAssembler<>::variable    variable;
typedef gsExprAssembler<>::space       space;
typedef gsExprAssembler<>::solution    solution;

int main(int argc, char *argv[])
{
    //////////////////
    // COMMAND LINE //
    //////////////////

    bool plot = false;
    index_t numRefine  = 5;
    index_t numElevate = 0;
    bool last = false;
    std::string fn("./double_poisson_mesh.xml");

    gsCmdLine cmd("Tutorial on solving two Poisson problems monolithically.");
    cmd.addInt("e", "degreeElevation",
               "Number of degree elevation steps to perform before solving (0: equalize degree in all directions)", numElevate );
    cmd.addInt("r", "uniformRefine", "Number of Uniform h-refinement loops",  numRefine );
    cmd.addString("f", "file", "Input XML file", fn );
    cmd.addSwitch("plot", "Create a ParaView visualization file with the solution", plot);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }


    ////////////////
    // FILE INPUT //
    ////////////////

    gsFileData<> fd(fn);
    gsInfo << "Loaded file "<< fd.lastPath() << std::endl;

    gsMultiPatch<> multiPatchDomain;
    fd.getId(0, multiPatchDomain); // id=0: Multipatch domain

    gsFunctionExpr<> source_p1;
    fd.getId(1, source_p1); // id=1: source function for problem 1
    gsInfo<<"Source function "<< source_p1 << "\n";

    gsBoundaryConditions<> bc_p1;
    fd.getId(2, bc_p1); // id=2: boundary conditions for problem 1
    bc_p1.setGeoMap(multiPatchDomain);
    gsInfo<<"Boundary conditions for problem 1:\n"<< bc_p1 << std::endl;

    gsBoundaryConditions<> bc_p2;
    fd.getId(3, bc_p2); // id=3: boundary conditions for problem 2
    bc_p2.setGeoMap(multiPatchDomain);
    gsInfo<<"Boundary conditions for problem 2:\n"<< bc_p2 << std::endl;

    gsOptionList Aopt;
    fd.getId(10, Aopt); // id=10: assembler options


    ////////////////
    // REFINEMENT //
    ////////////////

    gsMultiBasis<> dbasis(multiPatchDomain, true);//true: poly-splines (not NURBS)

    // Elevate and p-refine the basis to order p + numElevate
    // where p is the highest degree in the bases
    dbasis.setDegree( dbasis.maxCwiseDegree() + numElevate);

    // h-refine each basis
    for (int r =0; r < numRefine; ++r)
        dbasis.uniformRefine();
    numRefine = 0;

    gsInfo << "Patches: "<< multiPatchDomain.nPatches() <<", degree: "<< dbasis.minCwiseDegree() << std::endl;
#ifdef _OPENMP
    gsInfo<< "Available threads: "<< omp_get_max_threads() << std::endl;
#endif

    ///////////////////
    // PROBLEM SETUP //
    ///////////////////

    gsExprAssembler<> assembler(2,2);
    assembler.setOptions(Aopt);

    gsInfo << "Active options:\n" << assembler.options() << std::endl;

    // Elements used for numerical integration
    assembler.setIntegrationElements(dbasis);

    // Set the geometry map
    geometryMap geoMap = assembler.getMap(multiPatchDomain);

    // Set the discretization space for problem 1
    space space_p1 = assembler.getSpace(dbasis, 1, 0);
    // Set the discretization space for problem 1
    space space_p2 = assembler.getSpace(dbasis, 1, 1);

    // Set the source term
    auto s_p1 = assembler.getCoeff(source_p1, geoMap);

    // Solution vector and solution variable for problem 1
    gsMatrix<> solVector_p1;
    solution uSol_p1 = assembler.getSolution(space_p1, solVector_p1);

    // Solution vector and solution variable for problem 2
    gsMatrix<> solVector_p2;
    solution uSol_p2 = assembler.getSolution(space_p2, solVector_p2);


    ////////////
    // SOLVER //
    ////////////

    gsSparseSolver<>::CGDiagonal solver;

    gsInfo << "(dot1=assembled, dot2=solved)\n\nDoFs: ";
    double setup_time(0), ma_time(0), slv_time(0);
    gsStopwatch timer;

    // Set-up boundary conditions for problem 1
    space_p1.setup(bc_p1, dirichlet::l2Projection, 0);
    // Set-up boundary conditions for problem 2
    space_p2.setup(bc_p2, dirichlet::l2Projection, 0);

    // Initialize the system
    assembler.initSystem();
    setup_time += timer.stop();

    gsInfo << assembler.numDofs() << std::flush;

    timer.restart();
    // Compute the system matrix and right-hand side for problem 1
    assembler.assemble(
        igrad(space_p1, geoMap) * igrad(space_p1, geoMap).tr() * meas(geoMap) //matrix
        ,
        space_p1 * s_p1 * meas(geoMap) //rhs vector
    );
    // Compute the system matrix and right-hand side for problem 2
    assembler.assemble(
        igrad(space_p2, geoMap) * igrad(space_p2, geoMap).tr() * meas(geoMap) //matrix
    );

    // Compute the Neumann terms defined on physical space for problem 1
    auto g_N = assembler.getBdrFunction(geoMap);
    assembler.assembleBdr(bc_p1.get("Neumann"), space_p1 * g_N.tr() * nv(geoMap) );

    // Compute the Neumann terms defined on physical space for problem 2
    assembler.assembleBdr(bc_p2.get("Neumann"), space_p2 * g_N.tr() * nv(geoMap) );

    ma_time += timer.stop();

    // gsDebugVar(A.matrix().toDense());
    // gsDebugVar(A.rhs().transpose());

    gsInfo << "." << std::flush;// Assemblying done

    timer.restart();
    solver.compute( assembler.matrix() );
    gsMatrix<> fullSolVector = solver.solve(assembler.rhs());
    slv_time += timer.stop();

    gsInfo << "." << std::endl; // Linear solving done

    // extract solution of sub problems from fullSolVector
    const index_t full_size = fullSolVector.size();
    solVector_p1 = fullSolVector.block(0, 0, full_size/2, 1);
    solVector_p2 = fullSolVector.block(full_size/2, 0, full_size/2, 1);
    gsInfo << "Size of full solution vector: " << full_size << std::endl;
    gsInfo << "Size of solution vector for problem 1: " << solVector_p1.size() << std::endl;
    gsInfo << "Size of solution vector for problem 2: " << solVector_p2.size() << std::endl;


    /////////////
    // TIMINGS //
    /////////////

    timer.stop();
    gsInfo << "\n\nTotal time: " << setup_time+ma_time+slv_time << std::endl;
    gsInfo << "     Setup: " << setup_time << std::endl;
    gsInfo << "  Assembly: " << ma_time    << std::endl;
    gsInfo << "   Solving: " << slv_time   << std::endl;
    

    ////////////
    // EXPORT //
    ////////////

    if (plot)
    {
        gsInfo<<"Plotting in Paraview...\n";

        gsExprEvaluator<> ev(assembler);
        gsParaviewCollection collection("ParaviewOutput/solution", &ev);
        collection.options().setSwitch("plotElements", true);
        collection.options().setInt("plotElements.resolution", 16);
        collection.newTimeStep(&multiPatchDomain);
        collection.addField(uSol_p1,"numerical solution for problem 1");
        collection.addField(uSol_p2,"numerical solution for problem 2");
        collection.saveTimeStep();
        collection.save();


        gsFileManager::open("ParaviewOutput/solution.pvd");
    }
    else
        gsInfo << "Done. No output created, re-run with --plot to get a ParaView "
                  "file containing the solution.\n";

    return EXIT_SUCCESS;

}// end main
