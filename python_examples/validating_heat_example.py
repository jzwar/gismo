import gustaf as gus
import numpy as np

# Options
askew = True
dx = 1e-6
component_id = 3

if askew:
    geometry_spline = gus.Bezier(degrees=[2, 2], control_points=[
        [0.0, 0.0],
        [1.0, 0.5],
        [2.0, 0.2],
        [0.5, 1.5],
        [1.0, 1.5],
        [1.5, 1.5],
        [0.0, 3.0],
        [1.0, 2.5],
        [2.0, 3.0],
    ])
else:
    geometry_spline = gus.Bezier(degrees=[2, 2], control_points=[
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.5],
        [1.0, 1.5],
        [2.0, 1.5],
        [0.0, 3.0],
        [1.0, 3.0],
        [2.0, 3.0],
    ])

dgeom_dx = geometry_spline.copy()
dgeom_dx.control_points[component_id][0] += dx

if askew:
    solution_spline = gus.Bezier(degrees=[2, 2], control_points=[
        [0],
        [0],
        [0],
        [1.119908048140325],
        [1.996676895007331],
        [1.252151918795698],
        [3.77684486652888],
        [2.352393098252772],
        [3.736685523138077],
    ])
else:
    solution_spline = gus.Bezier(degrees=[2, 2], control_points=[
        [0.0],
        [0.0],
        [0.0],
        [1.5],
        [1.5],
        [1.5],
        [3.0],
        [3.0],
        [3.0],
    ])

evaluation_point = np.array([[0.25, 0.6]])

# Geometric expressions
jac = geometry_spline.jacobian(queries=evaluation_point)[0]
inv_jac = np.linalg.inv(jac)
jac_dx = dgeom_dx.jacobian(queries=evaluation_point)[0]
inv_jac_dx = np.linalg.inv(jac_dx)

# Gradient field of the solution
grad_sol = solution_spline.jacobian(queries=evaluation_point)[0]

# Basis function derivatives and the individual components for derivation
basis_f_deriv0 = solution_spline.basis_derivative_and_support(
    queries=evaluation_point, orders=[1, 0])[0]
basis_f_deriv1 = solution_spline.basis_derivative_and_support(
    queries=evaluation_point, orders=[0, 1])[0]
basis_f_derivs = np.vstack((
    basis_f_deriv0,
    basis_f_deriv1
))
dJdC0 = np.reshape(np.stack(
    [
        basis_f_deriv0,
        basis_f_deriv1,
        np.zeros(basis_f_deriv0.shape),
        np.zeros(basis_f_deriv0.shape),
    ], axis=2
), (1, basis_f_deriv0.shape[1], 2, 2))[0]
dJdC1 = np.reshape(np.stack(
    [
        np.zeros(basis_f_deriv0.shape),
        np.zeros(basis_f_deriv0.shape),
        basis_f_deriv0,
        basis_f_deriv1,
    ], axis=2
), (1, basis_f_deriv0.shape[1], 2, 2))[0]


# Components
BL0 = np.matmul(grad_sol, inv_jac)
BL1 = np.matmul(basis_f_derivs.T, inv_jac).T
BL2 = np.linalg.det(jac)

# Solution vector
bilinear_solution = np.matmul(BL0, BL1) * BL2

###
# Derivative of individual components
###
# FD solutions
BL0_dx = np.matmul(grad_sol, inv_jac_dx)
BL1_dx = np.matmul(basis_f_derivs.T, inv_jac_dx).T
BL2_dx = np.linalg.det(jac_dx)
# Auxiliary values
dJdC0_mapped = np.einsum("ijk,kl->ijl", dJdC0, inv_jac)


# First Component np.matmul(grad_sol, inv_jac)
dBL0_dx_fd = (BL0_dx - BL0) / dx
dBL0_dx = -np.einsum("i,kij->kj", BL0[0], dJdC0_mapped)
print(dBL0_dx)
print("\nComparison of values for component 0\n",
      dBL0_dx_fd[0], "\n", dBL0_dx[component_id],
      "\n are they equal : ",
      np.allclose(dBL0_dx_fd[0], dBL0_dx[component_id]), "\n")

# Second Component np.matmul(basis_f_derivs.T, inv_jac).T
dBL1_dx_fd = (BL1_dx - BL1) / dx
dBL1_dx = -np.einsum("ij,kil->jkl", BL1, dJdC0_mapped)
print("Comparison of values for component 2\n",
      dBL1_dx_fd, "\n", dBL1_dx[component_id].T,
      "\n are they equal : ",
      np.allclose(dBL1_dx_fd, dBL1_dx[component_id].T), "\n")

# Third Component np.linalg.det(jac)
dBL2_dx_fd = (BL2_dx - BL2) / dx
dBL2_dx = np.einsum("ijj->i", dJdC0_mapped) * BL2
print("Comparison of values for component 3\n",
      dBL2_dx_fd, "\n", dBL2_dx[component_id],
      "\n are they equal : ",
      np.allclose(dBL2_dx_fd, dBL2_dx[component_id]), "\n")


expr_total_0 = np.einsum("ij,jk->ik", dBL0_dx, BL1) * BL2
expr_total_2 = np.einsum("i,ij,k->jk", BL0[0], BL1, dBL2_dx)


# Checkers
print("Check")
