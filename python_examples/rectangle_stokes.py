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

boundary_conditions = {
    "tag": "boundaryConditions",
    "attributes": {"multipatch": "0", "id": "1"},
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
            "text":"4",
        },
        {
            "tag": "Function",
            "attributes": {
                "type": "FunctionExpr",
                "dim": "2",
                "index": "2",
            },
            "text":"y",
        },
        # {
        #     "tag": "Function",
        #     "attributes": {
        #         "type": "FunctionExpr",
        #         "dim": "2",
        #         "index": "2",
        #         "c": "2",
        #     },
        #     "children": [
        #         {
        #             "tag": "c",
        #             "attributes": {"index": "0"},
        #             "text": "0",
        #         },
        #         {
        #             "tag": "c",
        #             "attributes": {"index": "1"},
        #             "text": "-10000",
        #         },
        #     ],
        # },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "2",
                "unknown": "1",
                "component" : "0",
                "name": "BID1",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "0",
                "unknown": "1",
                "component" : "1",
                "name": "BID1",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "0",
                "unknown": "1",
                "component" : "1",
                "name": "BID2",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "0",
                "unknown": "1",
                "component" : "0",
                "name": "BID3",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "0",
                "unknown": "1",
                "component" : "1",
                "name": "BID3",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "1",
                "unknown": "1",
                "component" : "0",
                "name": "BID4",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": "0",
                "unknown": "1",
                "component" : "1",
                "name": "BID4",
            },
        },
    ],
}

gismo_options.append(assembly_options)
gismo_options.append(boundary_conditions)

# Define geometry
rectangle = gus.spline.create.box(3,4).bspline
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
gus.spline.io.gismo.export("rectangle_mesh.xml", multipatch=multipatch, options=gismo_options )

    
    
