import gustaf as gus
import numpy as np


my_ms = [gus.Bezier(degrees=[1,1], control_points=[[0,0],[1,0],[0,2],[1,2]])]

gus.show(my_ms)
# Use Multipatch as a basis for export
multipatch = gus.spline.splinepy.Multipatch(my_ms)
multipatch.boundaries_from_continuity()
gus.spline.io.gismo.export("rectangle_mesh.xml",multipatch=multipatch)


