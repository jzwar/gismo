import gustaf as gus
import numpy as np
import scipy
import subprocess
import os

gismo_options = [
    {
        "tag": "Function",
        "attributes": {"type": "FunctionExpr", "id": "1", "dim": "2"},
        "text": "\n    ",
        "children": [
            {
                "tag": "c",
                "attributes": {"index": "0"},
                "text": "0",
            },
            {
                "tag": "c",
                "attributes": {"index": "1"},
                "text": "0",
            },
        ],
    },
    {
        "tag": "boundaryConditions",
        "attributes": {"multipatch": "0", "id": "2"},
        "children": [
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": "2",
                    "index": "0",
                },
                "text": "0",
            },
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": "2",
                    "index": "1",
                },
                "text": "\n      -1\n    ",
            },
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": "2",
                    "index": "2",
                    "c": "2",
                },
                "children": [
                    {
                        "tag": "c",
                        "attributes": {"index": "0"},
                        "text": "0",
                    },
                    {
                        "tag": "c",
                        "attributes": {"index": "1"},
                        "text": "-10000",
                    },
                ],
            },
            {
                "tag": "bc",
                "attributes": {
                    "type": "Dirichlet",
                    "function": "0",
                    "unknown": "0",
                    "name": "BID2",
                },
            },
            {
                "tag": "bc",
                "attributes": {
                    "type": "Dirichlet",
                    "function": "0",
                    "unknown": "1",
                    "name": "BID2",
                },
            },
            {
                "tag": "bc",
                "attributes": {
                    "type": "Neumann",
                    "function": "2",
                    "unknown": "0",
                    "name": "BID6",
                },
            },
        ],
    },
    {
        "tag": "OptionList",
        "attributes": {"id": "3"},
        "text": "\n    ",
        "children": [
            {
                "tag": "int",
                "attributes": {
                    "label": "DirichletStrategy",
                    "desc": "Method for enforcement of Dirichlet BCs [11..14]",
                    "value": "11",
                },
            },
            {
                "tag": "int",
                "attributes": {
                    "label": "DirichletValues",
                    "desc": "Method for computation of Dirichlet DoF values [100..103]",
                    "value": "101",
                },
            },
            {
                "tag": "int",
                "attributes": {
                    "label": "InterfaceStrategy",
                    "desc": "Method of treatment of patch interfaces [0..3]",
                    "value": "1",
                },
            },
            {
                "tag": "real",
                "attributes": {
                    "label": "bdA",
                    "desc": "Estimated nonzeros per column of the matrix: bdA*deg + bdB",
                    "value": "2",
                },
            },
            {
                "tag": "int",
                "attributes": {
                    "label": "bdB",
                    "desc": "Estimated nonzeros per column of the matrix: bdA*deg + bdB",
                    "value": "1",
                },
            },
            {
                "tag": "real",
                "attributes": {
                    "label": "bdO",
                    "desc": "Overhead of sparse mem. allocation: (1+bdO)(bdA*deg + bdB) [0..1]",
                    "value": "0.333",
                },
            },
            {
                "tag": "real",
                "attributes": {
                    "label": "quA",
                    "desc": "Number of quadrature points: quA*deg + quB",
                    "value": "1",
                },
            },
            {
                "tag": "int",
                "attributes": {
                    "label": "quB",
                    "desc": "Number of quadrature points: quA*deg + quB",
                    "value": "1",
                },
            },
            {
                "tag": "int",
                "attributes": {
                    "label": "quRule",
                    "desc": "Quadrature rule [1:GaussLegendre,2:GaussLobatto]",
                    "value": "1",
                },
            },
        ],
    },
]

gus.settings.NTHREADS = 8
length = 2
height = 1
depth = 1/4
tiling = [8, 4, 1]
nthreads= 24
load_b_tiles = 1
para_degs = [0, 0, 0]
scaling_factor_objective_function = 100
design_vars = tiling # [8, 4, 1]


filename = ("lattice_structure_" + str(tiling[0]) + "x" + str(tiling[1]) + "x" + str(tiling[2]) + ".xml")
log_file = "log_file.tsv"

def prepare_microstructure(parameters):
    generator = gus.spline.microstructure.Microstructure()
    deformation_function = gus.spline.create.box(length, height, depth).bspline
    for i, t in enumerate(tiling):
        if t < 2:
            continue
        deformation_function.insert_knots(i, [j * (1 / t) for j in range(1, t)])

    generator.deformation_function = deformation_function
    generator.tiling=[1,1,1]
    generator.microtile = gus.spline.microstructure.tiles.Cube3D()
    
    # Create Parameter spline
    knot_vectors = []
    for i, (p, d) in enumerate(zip(para_degs,design_vars)):
        i_knots = [j * (1/(d - p)) for j in range(1, d - p)]
        knot_vectors.append(
            [0] * (p + 1) + i_knots + [1] * (p + 1)
        )
    
    parameter_spline = gus.BSpline(
        degrees=para_degs,
        knot_vectors=knot_vectors,
        control_points=parameters.reshape(-1,1)
    )

    def foo(x):
        """
        Parametrization Function (determines thickness)
        """
        return parameter_spline.evaluate(x)

    def foo_deriv(x):
        basis_function_matrix = np.zeros((
            x.shape[0],
            parameter_spline.control_points.shape[0]
        ))
        basis_functions, support = parameter_spline.basis_and_support(x)
        np.put_along_axis(basis_function_matrix, support, basis_functions, axis=1)
        return basis_function_matrix.reshape(x.shape[0], 1, -1)

    generator.parametrization_function = foo
    generator.parameter_sensitivity_function = foo_deriv
    my_ms, my_ms_der = generator.create(contact_length=0.3)

    def identifier_function(deformation_function, face_id):
        boundary_spline = deformation_function.extract_boundaries(face_id)[0]

        def identifier_function(x):
            distance_2_boundary = boundary_spline.proximities(
                queries=x, initial_guess_sample_resolutions=[4,4], tolerance=1e-9
            )[3]
            return distance_2_boundary.flatten() < 1e-8

        return identifier_function

    def identifier_function_neumann(x):
        return (x[:,0] >= (tiling[0] - load_b_tiles) / tiling[1] * length-1e-12)


    multipatch = gus.spline.splinepy.Multipatch(my_ms)
    multipatch.add_fields(*my_ms_der, check_compliance=True, check_conformity=True)
    multipatch.determine_interfaces()
    for i in range(deformation_function.dim * 2):
        multipatch.boundary_from_function(
            identifier_function(generator.deformation_function, i)
        )
    multipatch.boundary_from_function(
        identifier_function_neumann, mask=[5]
    )
    gus.show(multipatch.splines, control_points=False, knots=False, resolutions=3)
    gus.spline.io.gismo.export(filename, multipatch=multipatch, options=gismo_options, export_fields=True)

def read_jacobians():
    jacs =np.genfromtxt(fname="sensitivities.out")
    return jacs

def read_objective_function():
    obj_val =float(np.genfromtxt(fname="objective_function.out"))
    return obj_val

def run_gismo(sensitivities=False, plot=False, refinement=None):
    process_call = [
          "./linear_elasticity_expressions",
          "-f",
          filename, 
          "-p", 
          str(nthreads),
          "--compute-objective-function",
          "--output-to-file"
        ]
    if sensitivities:
        process_call += [
          "--compute-sensitivities",
          "-x", 
          filename + ".fields.xml"]
    if plot:
        process_call += [
            "--plot"
        ]
    if refinement is not None:
        process_call += [
          "-r",
          str(refinement),
          ]
    text = subprocess.run(process_call,
        capture_output=True,encoding="ascii")
    return text.returncode

def evaluate_iteration(x):
    prepare_microstructure(x)
    run_gismo(sensitivities=True, refinement=1)
    return read_objective_function() * scaling_factor_objective_function

def evaluate_jacobian(x):
    prepare_microstructure(x)
    run_gismo(sensitivities=True, refinement=1)
    with open(log_file, "a") as log:
        log.write(
                str(read_objective_function()) 
                + " " 
                + " ".join(str(xx) for xx in (read_jacobians() * 100).tolist()
                           )
                )
    return read_jacobians() * scaling_factor_objective_function

def call_back_optimization(x):
    print(f"{read_objective_function()} " + "  ".join(str(xx) for xx in x.tolist()))

def main():
    n_design_vars = np.prod(design_vars)
    initial_guess = np.ones((n_design_vars, 1)) * 0.30

    # Mass constraint
    A = np.ones((1, n_design_vars))
    c = 0
    d = n_design_vars * 0.30
    C2 = scipy.optimize.LinearConstraint(A, c, d)

    optim = scipy.optimize.minimize(
        evaluate_iteration,
        initial_guess,
        method='SLSQP',
        jac=evaluate_jacobian,
        bounds = [(0.01,0.49) for _ in range(n_design_vars)],
        constraints=C2,
        options={'disp': True},
        callback = call_back_optimization
    )
    # Finalize
    prepare_microstructure(optim.x)
    run_gismo(sensitivities=False, plot=True, refinement=True)
    print("Best Parameters : ")
    print(optim.x)
    print(optim)
    

if __name__ == "__main__":
    main()
