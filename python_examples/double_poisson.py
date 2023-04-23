import gustaf as gus
import numpy as np


gismo_options = []

assembly_options = {
    "tag": "OptionList",
    "attributes": {"id": "10"},
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
}

source_term_p1 = {
    "tag": "Function",
    "attributes": {
        "id": "1",
        "dim": "2",
    },
    "text": "2*pi^2*sin(pi*x)*sin(pi*y)"
}

boundary_conditions_p1 = {
    "tag": "boundaryConditions",
    "attributes": {
        "id": "2",
        "multipatch": "0"
    },
    "children": [
        {
            "tag": "Function",
            "attributes": {
                "type": "FunctionExpr",
                "dim": "2",
                "index": "0",
            },
            "text": "sin(pi*x)*sin(pi*y)",
        },
        {
            "tag": "Function",
            "attributes": {
                "type": "FunctionExpr",
                "dim": "2",
                "index": "1",
                "c": "2",
            },
            "children": [
                {
                    "tag": "c",
                    "attributes": {"index": "0"},
                    "text": "pi*cos(pi*x)*sin(pi*y)",
                },
                {
                    "tag": "c",
                    "attributes": {"index": "1"},
                    "text": "pi*sin(pi*x)*cos(pi*y)",
                },
            ],
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "0",
                "unknown": "0",
                "name": "BID1",
            },
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
                "type": "Neumann",
                "function": "1",
                "unknown": "0",
                "name": "BID3",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Neumann",
                "function": "1",
                "unknown": "0",
                "name": "BID4",
            },
        },
    ],
}

boundary_conditions_p2 = {
    "tag": "boundaryConditions",
    "attributes": {
        "id": "3",
        "multipatch": "0"
    },
    "children": [
        {
            "tag": "Function",
            "attributes": {
                "type": "FunctionExpr",
                "dim": "2",
                "index": "0",
            },
            "text": "x+y",
        },
        {
            "tag": "Function",
            "attributes": {
                "type": "FunctionExpr",
                "dim": "2",
                "index": "1",
                "c": "2",
            },
            "children": [
                {
                    "tag": "c",
                    "attributes": {"index": "0"},
                    "text": "1",
                },
                {
                    "tag": "c",
                    "attributes": {"index": "1"},
                    "text": "1",
                },
            ],
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "0",
                "unknown": "0",
                "name": "BID1",
            },
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
                "type": "Neumann",
                "function": "1",
                "unknown": "0",
                "name": "BID3",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Neumann",
                "function": "1",
                "unknown": "0",
                "name": "BID4",
            },
        },
    ],
}

gismo_options.append(assembly_options)
gismo_options.append(source_term_p1)
gismo_options.append(boundary_conditions_p1)
gismo_options.append(boundary_conditions_p2)

# Define geometry
rectangle = gus.spline.create.box(1,1).bspline
rectangle.elevate_degrees([0,1])

# refine and extract mp
# rectangle.elevate_degrees([0,1])
# rectangle.insert_knots(0, [.5])
# rectangle.insert_knots(1, [.5])
multipatch = gus.spline.splinepy.Multipatch(rectangle.extract.beziers())
gus.show(multipatch.splines)

# Define boundaries
multipatch.boundaries_from_continuity()

# Export
gus.spline.io.gismo.export(
    "double_poisson_mesh.xml", multipatch=multipatch, options=gismo_options 
)

    
    
