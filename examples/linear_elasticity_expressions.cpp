/** @file poisson2_example.cpp

    @brief Tutorial on how to use expression assembler to solve the Poisson
   equation

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): A. Mantzaflaris
*/

//! [Include namespace]
#include <gismo.h>

using namespace gismo;
//! [Include namespace]

int main(int argc, char *argv[]) {
  ////////////////////////////////
  // Parse Command Line Options //
  ////////////////////////////////

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

  std::string fn("pde/poisson2d_bvp.xml");
  cmd.addString("f", "file", "Input XML file", fn);

  // A few more mesh options
  int mp_id{0}, source_id{1}, bc_id{2}, ass_opt_id{4};
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
  gsInfo << "Mulipatch geometry set...\n";

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

  // Elevate and p-refine the basis to order p + numElevate
  // where p is the highest degree in the bases
  // function_basis.setDegree(function_basis.maxCwiseDegree() + numRefine);

  // h-refine each basis
  for (int r = 0; r < numRefine; ++r) function_basis.uniformRefine();

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

  typedef gsExprAssembler<>::geometryMap geometryMap;
  typedef gsExprAssembler<>::variable variable;
  typedef gsExprAssembler<>::space space;
  typedef gsExprAssembler<>::solution solution;

  // Elements used for numerical integration
  expr_assembler.setIntegrationElements(function_basis);
  gsExprEvaluator<> expression_evaluator(expr_assembler);

  // Set the geometry map
  geometryMap G = expr_assembler.getMap(mp);

  // Set the discretization spaces
  const int solution_field_dimension{2};
  space u = expr_assembler.getSpace(function_basis, solution_field_dimension);

  // Set the source term
  auto ff = expr_assembler.getCoeff(f, G);

  // Solution vector and solution variable
  gsMatrix<> solVector;
  solution u_sol = expr_assembler.getSolution(u, solVector);

  //! [Problem setup]

  //! [Solver loop]
  gsSparseSolver<>::LU solver;

  double setup_time(0), ma_time(0), slv_time(0), err_time(0);
  gsStopwatch timer;

  //        u.setup(bc, dirichlet::interpolation, 0);
  u.setup(bc, dirichlet::l2Projection, 0);

  // Initialize the system
  expr_assembler.initSystem();
  setup_time += timer.stop();

  gsInfo << expr_assembler.numDofs() << std::flush;

  timer.restart();

  // Compute the system matrix and right-hand side
  auto bilin_lambda = lame_lambda * idiv(u, G) * idiv(u, G).tr();

  // NOTE : Tensor contraction A:B can be written as trace(A^T * B);
  auto phys_jacobian = ijac(u, G);
  auto bilin_mu_1 = lame_mu * (phys_jacobian % phys_jacobian.tr());
  auto bilin_mu_2 = lame_mu * (phys_jacobian.cwisetr() % phys_jacobian.tr());

  auto bilin = (bilin_lambda + bilin_mu_1 + bilin_mu_2) * meas(G);

  expr_assembler.assemble(bilin  // matrix
                          ,
                          rho * u * ff * meas(G)  // rhs vector
  );

  // Compute the Neumann terms defined on physical space
  auto g_N = expr_assembler.getBdrFunction(G);
  expr_assembler.assembleBdr(bc.get("Neumann"), u * g_N.tr() * nv(G));

  ma_time += timer.stop();

  gsInfo << "." << std::flush;  // Assemblying done

  timer.restart();
  solver.compute(expr_assembler.matrix());
  solVector = solver.solve(expr_assembler.rhs());
  slv_time += timer.stop();

  gsInfo << "." << std::flush;  // Linear solving done

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
    auto d_jac_d_c = jac(u);  // assumes isoparametrics
    auto inv_jac = jac(G).inv();
    auto d_jac_inv_d_c = -inv_jac * d_jac_d_c;

    // Differntiating lambda
    auto bilin_lambda_der1 =
        (d_jac_inv_d_c * ijac(u_sol, G)).trace() * idiv(u, G).tr();
    // auto bilin_lambda_der2 =
    //     idiv(u_sol, G) * (d_jac_inv_d_c * igrad(u, G).tr());
    auto bilin_lambda_der = lame_lambda * (bilin_lambda_der1) * meas(G);

    auto derivative = bilin_lambda_der;

    gsInfo << "\nFirst Expression:\n"
           << evaluator.eval(
                  (d_jac_inv_d_c * ijac(u_sol, G)).trace() * idiv(u, G).tr(),
                  evalPoint)
           << "\n";

    timer.stop();
    gsInfo << "\n\nTotal time: " << setup_time + ma_time + slv_time + err_time
           << "\n";
    gsInfo << "     Setup: " << setup_time << "\n";
    gsInfo << "  Assembly: " << ma_time << "\n";
    gsInfo << "   Solving: " << slv_time << "\n";
  }
  //! [Export visualization in ParaView]
  if (plot) {
    gsInfo << "Plotting in Paraview...\n";
    expression_evaluator.options().setSwitch("plot.elements", true);
    expression_evaluator.writeParaview(u_sol, G, "solution");

    gsFileManager::open("solution.pvd");
  } else {
    gsInfo << "Done. No output created, re-run with --plot to get a ParaView "
              "file containing the solution.\n";
  }
  //! [Export visualization in ParaView]

  if (export_xml) {
    gsInfo << "Writing to G+Smo XML." << std::flush;
    gsMultiPatch<> mpsol;
    gsMatrix<> full_solution;

    gsFileData<> output;

    output << solVector;

    u_sol.extractFull(full_solution);
    output << full_solution;
    output.save("solution-field.xml");
  }

  return EXIT_SUCCESS;

}  // end main
