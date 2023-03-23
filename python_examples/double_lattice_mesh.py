import gustaf as gus

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
                    "name": "BID1",
                },
            },
            {
                "tag": "bc",
                "attributes": {
                    "type": "Dirichlet",
                    "function": "0",
                    "unknown": "1",
                    "name": "BID1",
                },
            },
            {
                "tag": "bc",
                "attributes": {
                    "type": "Neumann",
                    "function": "2",
                    "unknown": "0",
                    "name": "BID3",
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

generator = gus.spline.microstructure.Microstructure()
generator.deformation_function = gus.Bezier(
    degrees=[1, 1],
    control_points=[[0, 0], [1, 0], [0, 1], [1, 1]],
)
generator.microtile = gus.spline.microstructure.tiles.DoubleLatticeTile()

# Test Jacobians
generator.tiling = [1, 1]
my_ms = generator.create(contact_length=0.5)

generator.show(
    knots=True, control_points=False, title="2D Lattice Microstructure"
)


def identifier_function(deformation_function, face_id):
    boundary_spline = deformation_function.extract_boundaries(face_id)[0]

    def identifier_function(x):
        distance_2_boundary = boundary_spline.proximities(
            queries=x, initial_guess_sample_resolutions=4, tolerance=1e-9
        )[3]
        return distance_2_boundary.flatten() < 1e-8

    return identifier_function


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

gus.spline.io.gismo.export("lattice_1_mesh.xml", multipatch=multipatch, gismo_options=gismo_options)
