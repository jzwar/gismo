import gustaf as gus
import numpy as np


para_s = gus.Bezier(
    degrees=[1,1],
    control_points=(np.random.rand(4,1)-.5) * 0.1 + .125,
)

deformation_function = gus.Bezier(
    degrees=[1, 1], control_points=[[0, 0], [2, 0], [0, 1], [2, 1]]
)

def boundary_identifier_function(ID, tolerance=0.001):
    boundary_spline = deformation_function.extract_boundaries([ID])[0]
    def BI_function(x):
        return (boundary_spline.proximities(
            x,
            initial_guess_sample_resolutions=3,
                )[3] < tolerance).flatten()
    return BI_function

def parameter_function_double_lattice(x):
    """
    Parametrization Function (determines thickness)
    """
    return tuple([para_s.evaluate(x).flatten()])


# Test new microstructure
generator = gus.spline.microstructure.Microstructure()
# outer geometry
generator.deformation_function = deformation_function
generator.microtile = gus.spline.microstructure.tiles.DoubleLatticeTile()

# how many structures should be inside the cube
generator.tiling = [10,5]
generator.parametrization_function = parameter_function_double_lattice
my_ms = generator.create(
    # closing_face='y',
    contact_length=0.4)

gus.show(my_ms, knots=True, control_points=False, resolution=2)

# Use Multipatch as a basis for export
multipatch = gus.spline.splinepy.Multipatch(my_ms)
multipatch.boundary_from_function(boundary_identifier_function(0))
multipatch.boundary_from_function(boundary_identifier_function(1))
multipatch.boundary_from_function(boundary_identifier_function(2))
multipatch.boundary_from_function(boundary_identifier_function(3))

print( "All are positive : ",np.all([np.sign(np.linalg.det(t.jacobian([[.5,.5]]))) > 0
       for t in multipatch.splines]))

gus.spline.io.gismo.export("clean_testmesh.xml",multipatch=multipatch)


