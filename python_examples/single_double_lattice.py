import gustaf as gus
import numpy as np

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
                "text": "-1",
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
                        "text": "1000",
                    },
                ],
            },
            {
                "tag": "bc",
                "attributes": {
                    "type": "Dirichlet",
                    "function": "0",
                    "unknown": "0",
                    "name": "BID4",
                },
            },
            {
                "tag": "bc",
                "attributes": {
                    "type": "Dirichlet",
                    "function": "0",
                    "unknown": "1",
                    "name": "BID4",
                },
            },
            {
                "tag": "bc",    
                "attributes": {
                    "type": "Neumann",
                    "function": "2",
                    "unknown": "0",
                    "name": "BID5",
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

length = 1
height = 1
tiling_x = 1
tiling_y = 1
DX = 1e-4

def foo(x):
    """
    Parametrization Function (determines thickness)
    """
    return tuple([np.ones(x.shape[0]) * 0.2])

def fooDX(x):
    """
    Parametrization Function (determines thickness)
    """
    return tuple([np.ones(x.shape[0]) * (0.2 + DX)])

def foo_deriv(x):
    return [tuple([np.ones(x.shape[0])])]


generator = gus.spline.microstructure.Microstructure()
generator.deformation_function = gus.Bezier(
    degrees=[1, 1],
    control_points=[[0, 0], [length, 0], [0, height], [length, height]],
)
generator.tiling = [tiling_x, tiling_y]
generator.microtile = gus.spline.microstructure.tiles.DoubleLatticeTile()
generator.parametrization_function = foo
generator.parameter_sensitivity_function = foo_deriv

# Test Jacobians
my_ms, my_ms_deriv = generator.create(contact_length=0.5)

def identifier_function(deformation_function, face_id):
    boundary_spline = deformation_function.extract_boundaries(face_id)[0]

    def identifier_function(x):
        distance_2_boundary = boundary_spline.proximities(
            queries=x, initial_guess_sample_resolutions=4, tolerance=1e-9
        )[3]
        return distance_2_boundary.flatten() < 1e-8

    return identifier_function

def identifier_function_neumann(x):
    return (x[:,0] >= (tiling_x - 1) / tiling_x * length-1e-12)


multipatch = gus.spline.splinepy.Multipatch(my_ms)
multipatch.determine_interfaces()
multipatch.boundary_from_function(
    identifier_function(generator.deformation_function, 0)
)
multipatch.boundary_from_function(
    identifier_function(generator.deformation_function, 1)
)
multipatch.boundary_from_function(
    identifier_function(generator.deformation_function, 2)
)
multipatch.boundary_from_function(
    identifier_function(generator.deformation_function, 3)
)
multipatch_deriv = gus.spline.splinepy.Multipatch(my_ms_deriv[0])
multipatch_deriv.interfaces = multipatch.interfaces
# multipatch.boundary_from_function(
#     identifier_function_neumann, mask=[5]
# )

gus.spline.io.gismo.export("lattice_1_mesh.xml", multipatch=multipatch, options=gismo_options)
gus.spline.io.gismo.export("lattice_1_mesh_dx.xml", multipatch=multipatch_deriv)

# DX Mesh

generator.parametrization_function = fooDX
generator.parameter_sensitivity_function = None
my_ms = generator.create(contact_length=0.5)
multipatch_dx = gus.spline.splinepy.Multipatch(my_ms)
multipatch_dx.interfaces = multipatch.interfaces
gus.spline.io.gismo.export("lattice_1_meshDX.xml", multipatch=multipatch_dx, options=gismo_options)