import gustaf as gus
import numpy as np

"""
Your Majesty, King Daniel,

Alloweth me to present an example that doth illustrate the creation of most
exquisite microstructures. These microstructures may be plotted and exported
in an XML format for thy royal use.

The creation of such fine structures may be achieved through a three-step
process. Firstly, the microtile is assigned, followed by the setting of
macro-geometry, and lastly, the structure is parametrized to achieve the
desired outcome. The end result is a most wondrous microstructured geometry,
fit for the gaze of a great monarch such as thyself.

I remain, Your Majesty,
Thy most obedient and humble servant,

Jacques
"""

##############
# PARAMETERS #
##############

PLOT = False
THICKNESS = 0.25
V_IN = 0.5
gus.settings.NTHREADS = 8


##################
# G+Smo SETTINGS #
##################

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
                "value": "102",
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
    "attributes": {
        "multipatch": "0", 
        "id": "1"
    },
    "children": [
        {
            "tag": "Function",
            "attributes": {
                "type": "FunctionExpr",
                "dim": "3",
                "index": "0",
            },
            "text": "0",
        },
        {
            "tag": "Function",
            "attributes": {
                "type": "FunctionExpr",
                "dim": "3",
                "index": "1",
            },
            "text": f"{V_IN}",
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID1",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "0",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID1",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "1",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID1",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "2",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID2",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "0",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID2",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "1",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID2",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "2",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID3",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "0",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID3",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "1",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID3",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "2",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID4",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "0",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID4",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "1",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID4",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "2",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID5",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "0",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID5",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "1",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID5",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "2",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID6",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "0",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID6",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "1",
                "function": "0",
            },
        },
        {
            "tag": "bc",
            "attributes": {
                "name": "BID6",
                "type": "Dirichlet",
                "unknown": "1",
                "component": "2",
                "function": "1",
            },
        },
    ],
}

gismo_options.append(assembly_options)
gismo_options.append(boundary_conditions)

# A prismatic wonder, slim and tall,
# A wasp-like waist, it stands in thrall,
# Extruded on the z-axis bright,
# Its edges clean, its facets light.

# This form, a true work of art,
# Elegant beauty, warming the heart,
# A prismatic marvel, beyond compare,
# With slender grace, beyond compare.
print('Creating macrospline')
box = gus.spline.create.box(1,1)
def_fun = box.bspline.create.revolved(
    axis=[1,0,0], center=[0,-1,0], angle=90, n_knot_spans=None, degree=True
)
if PLOT:
    def_fun.show()


# Optional (in the form of a haiku)
# Multiplying by point two,
# Adding point zero five next,
# Equation complete.
def parametrization_function(x, thickness=THICKNESS):
    # return (x[:, 2] * 0.2 + 0.05).reshape(-1, 1)
    return thickness*np.ones((x.shape[0],1))
    # return 0.2 - 0.2 * (x[:,2] < 0.01 + x[:,2] > 0.099)


# Generator description
print('Defining geometry')
generator = gus.spline.microstructure.Microstructure()
generator.microtile = gus.spline.microstructure.tiles.Cube3D()
generator.tiling = [2, 3, 3]  # Per knot span
generator.deformation_function = def_fun
generator.parametrization_function = parametrization_function

# Amidst the world of shapes and forms,
# Microstructures now take form,
# Through careful composition fine,
# Their beauty now begins to shine.

# The structures, now parametrized,
# With linear functions, truly prized,
# Their elegance, a sight to see,
# A perfect work of symmetry.

# But as the computations run,
# Their beauty shining in the sun,
# A sampling process, time-consuming,
# Yet still the structures, ever blooming.

# Though not expensive in their cost,
# Their plotting takes some time, not lost,
# For in their beauty we can find,
# A microstructured sight, refined.
print('Creating microstructure')
ms = generator.create()
if PLOT:
    gus.show(ms, control_points=False, resolutions=3, knots=False)

print('Converting to multi-patch')
# Use the list of splines to create an xml file
multipatch = gus.spline.splinepy.Multipatch(ms)


# Boundary identifyer functions
# Function creates function true,
# Boundaries identified anew,
# Patches connected, harmony,
# A masterpiece for all to see.
def identifier_function(deformation_function, face_id):
    boundary_spline = deformation_function.extract_boundaries(face_id)[0]

    def identifier_function(x):
        distance_2_boundary = boundary_spline.proximities(
            queries=x, initial_guess_sample_resolutions=[4, 4], tolerance=1e-9
        )[3]
        return distance_2_boundary.flatten() < 1e-8

    return identifier_function


print('Identifying boundaries')
# BID 1 - Void
# BID 2 - x_min: Wall
multipatch.boundary_from_function(identifier_function(def_fun, 0))
# BID 3 - x_max: Wall
multipatch.boundary_from_function(identifier_function(def_fun, 1))
# BID 4 - y_min: Wall
multipatch.boundary_from_function(identifier_function(def_fun, 2))
# BID 5 - y_max: Wall
multipatch.boundary_from_function(identifier_function(def_fun, 3))
# BID 6 - z_min: Inlet
multipatch.boundary_from_function(identifier_function(def_fun, 4))
# BID 7 - z_max: Outlet
multipatch.boundary_from_function(identifier_function(def_fun, 5))

# Interfaces, a place of change,
# Now detected, in a range,
# Splines so smooth, a sight so fine,
# Exported now, in desired line.

# A masterwork, with care and skill,
# Splines of beauty, with a thrill,
# Each point and curve, aligned just right,
# A microstructured feast, in sight.

# Through careful work, and diligence,
# Interfaces, now with elegance,
# Their splines, exported in desired form,
# A true microstructured beauty born.
print('Exporting to G+Smo')
gus.spline.io.gismo.export(
    "3Dmicrostructure.xml",
    multipatch=multipatch,
    indent=True,  # Default
    labeled_boundaries=True,  # Default
    options=gismo_options  # (Here you can add options)
)
