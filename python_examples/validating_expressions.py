import gustaf as gus
import numpy as np


# Helper function for integration
def integrate_function(spline_function, order=None, dim=None, interval=[0, 1]):
    if dim is None or order is None:
        raise ValueError()

    # Get Legendre points
    positions, weights = np.polynomial.legendre.leggauss(order)
    positions = interval[0] + (interval[1] - interval[0]) / 2 * (positions + 1)
    weights = weights * (interval[1] - interval[0]) / 2
    positions = np.reshape(
        np.meshgrid(*[positions for _ in range(dim)]),
        (dim, -1)
    ).T
    weights = np.prod(
        np.reshape(
            np.meshgrid(*[weights for _ in range(dim)]),
            (dim, -1)
        ).T,
        axis=1
    )

    # Evaluate function
    function_values = spline_function(positions)
    return np.matmul(function_values.T, weights)


# Check the derivative of the determinante
"""
idea : d(det(J))/d(dC) = det(J) tr(J^{-1} dJdC)
"""

# Create some random spline
spline = gus.Bezier(
    degrees=[2, 1],
    control_points=[
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 5],
        [2, 1],
    ] + np.random.rand(6, 2)
)

# integrate Volume


def volume_function(queries):
    return np.linalg.det(spline.jacobian(queries=queries))


integrated_volume = integrate_function(volume_function, 10, 2)
print("The Integral over the surface is : ", integrated_volume)


def dJacdC(queries, dim):
    number_of_samples = queries.shape[0]
    number_of_ctps = spline.control_points.shape[0]

    basis_f_derivs0 = spline.basis_derivative_and_support(
        queries=queries, orders=[1, 0])[0]
    basis_f_derivs1 = spline.basis_derivative_and_support(
        queries=queries, orders=[0, 1])[0]
    if dim == 0:
        return np.reshape(np.stack(
            [
                basis_f_derivs0,
                basis_f_derivs1,
                np.zeros(basis_f_derivs1.shape),
                np.zeros(basis_f_derivs1.shape),
            ], axis=2
        ), (number_of_samples, number_of_ctps, 2, 2))
    if dim == 1:
        return np.reshape(np.stack(
            [
                np.zeros(basis_f_derivs1.shape),
                np.zeros(basis_f_derivs1.shape),
                basis_f_derivs0,
                basis_f_derivs1,

            ], axis=2
        ), (number_of_samples, number_of_ctps, 2, 2))


def volume_sensitivities(dim):
    def function(queries):
        jacs = spline.jacobian(queries=queries)
        inv_jacs = np.linalg.inv(jacs)
        jac_dets = np.linalg.det(jacs)
        dJdC = dJacdC(queries, dim)

        trace_of_J_dJdc = np.einsum('...ij,...mji', inv_jacs, dJdC)

        # Multiply with det jac ad return
        return np.einsum('ij,i->ij', trace_of_J_dJdc, jac_dets)
    return function

# Now to the approximation


def approximate_solution(der_var):
    sensitivities = []
    step_width = 0.01
    for i in range(spline.control_points.shape[0]):
        sp_c = spline.copy()
        sp_c.control_points[i, der_var] += step_width

        def d_volume_function(queries):
            return np.linalg.det(sp_c.jacobian(queries=queries))

        sensitivities.append(
            (integrate_function(d_volume_function, 10, 2) -
             integrated_volume) / step_width
        )
    return sensitivities


integrated_sensitivities_0 = integrate_function(volume_sensitivities(0), 10, 2)
print("The integrated sensitivity is : \t", integrated_sensitivities_0)
print("The approximated sensitivities are : \t", approximate_solution(0))

integrated_sensitivities_1 = integrate_function(volume_sensitivities(1), 10, 2)
print("The integrated sensitivity is : \t", integrated_sensitivities_1)
print("The approximated sensitivities are : \t", approximate_solution(1))


print("\nRatio between approx and analytical solution for deriv in xi : ", approximate_solution(0)
      / integrated_sensitivities_0
      * (np.abs(integrated_sensitivities_0) > 0.0001))
print("Ratio between approx and analytical solution for deriv in eta : ", approximate_solution(1)
      / integrated_sensitivities_1
      * (np.abs(integrated_sensitivities_1) > 0.0001))

print("Testing the expression on the geomtry from rectangle mesh.")
spline = gus.spline.io.gismo.load("./playground/rectangle_mesh.xml")[0]

# Validating the expressions from heat-equation
eval_point = np.array([[0.25, 0.6]])
djacdc0 = dJacdC(eval_point, 0)
djacdc1 = dJacdC(eval_point, 1)
print("dJacdC0 \n", djacdc0)
print("dJacdC1 \n", djacdc1)
jacs = spline.jacobian(queries=eval_point)
print(np.allclose(jacs, [[[2, 0], [0, 3]]]))
inv_jacs = np.linalg.inv(jacs)
jac_dets = np.linalg.det(jacs)
print("Jacs : \n", jacs)
print("Inv-Jacs : \n", inv_jacs)
trace_of_J_dJdc0 = np.einsum('ijk,ilkm->ilkm', inv_jacs, djacdc0)
trace_of_J_dJdc1 = np.einsum('ijk,ilkm->ilkm', inv_jacs, djacdc1)

print("\n\ndJdC multiplied by inv jacs:\n", trace_of_J_dJdc0)
print("\nTraces:\n", np.einsum("ilkk", trace_of_J_dJdc0))
print("\nTraces:\n", np.einsum("ilkk", trace_of_J_dJdc1))
