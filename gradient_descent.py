import numpy as np

def gradient_descent_exceeded_iterations(cnt_iter : int, max_iter : int):
    """Determines if the algorithm exceeded maximum iteration count.

    Args:
        cnt_iter (int): Iteration counter.
        max_iter (int): Maximum number of iterations.

    Returns:
        bool: Did the iteration counter exceed the maximum number of iterations?
    """
    return cnt_iter > max_iter

def gradient_descent_convergence_criterion(norm_gfval_k : float, eps : float):
    """Determines if the algorithm attained the desired tolerance in satisfying the convergence conditions.

    Args:
        norm_gfval_k (float): Norm of the gradient.
        eps (float): Epsilon tolerance.

    Returns:
        bool: Is the norm of the gradient smaller than the tolerance?
    """
    return norm_gfval_k < eps

def gradient_descent_print_header():
    """Prints the header of the optimization textual output.
    """
    print("{:<7}".format("Iter.") + " | " + "{:^10}".format("f(x)") + " | " + "{:^10}".format("||df(x)||") + " | " + "{:^10}".format("||d||"))

def gradient_descent_print_iteration(cnt_iter : int, fval_k : float, norm_gfval_k : float, step_size_k : float):
    """Prints the information of a single iteration of the gradient descent algorithm.

    Args:
        cnt_iter (int): Iteration counter.
        fval_k (float): Function value at current iterate.
        norm_gfval_k (float): Gradient norm value at current iterate.
        step_size_k (float): Size of the step at current iterate.
    """
    print(("%7d" % cnt_iter) + " | " + ("%.4e" % fval_k) + " | " + ("%.4e" % norm_gfval_k) + " | " + ("%.4e" % step_size_k))

def gradient_descent_print_output(cnt_iter : int, fval_k : float, norm_gfval_k : float, eps : float, max_iter : int):
    """Prints the final message of the search.

    Args:
        cnt_iter (int): Iteration counter.
        fval_k (float): Function value at current iterate.
        norm_gfval_k (float): Gradient norm value at current iterate.
        eps (float): Tolerance on the first order optimality condition for convergence.
        max_iter (int): Maximum number of iterations to perform.
    """
    # If the algorithm converged
    if gradient_descent_convergence_criterion(norm_gfval_k=norm_gfval_k, eps=eps):
        print()
        print("Gradient descent successfully converged in %d iterations." % cnt_iter)
        print("Optimal function value: %.4e." % fval_k)
        print("Optimality conditions : %.4e." % norm_gfval_k)

    # If the algorithm exceeded the iterations
    if gradient_descent_exceeded_iterations(cnt_iter=cnt_iter, max_iter=max_iter):
        print()
        print("Gradient descent exceeded the maximum number (%d) of iterations." % max_iter)
        print("Current function value: %.4e." % fval_k)
        print("Current optimality conditions : %.4e." % norm_gfval_k)


def gradient_descent(f, gf, alpha : float, x0 : np.ndarray, eps=1e-4, max_iter=1e3):
    """ Performs the constant step-length gradient descent optimization algorithm on the function f.

    Args:
        f (function): Function handle of the function to minimize.
        gf (function): Function handle of the gradient of the function to minimize.
        alpha (float): Constant step size.
        x0 (np.ndarray): Initial guess for the algorithm.
        eps (float, optional): Tolerance on the first order optimality condition for convergence. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations to perform. Defaults to 1e3.

    Returns:
        x: Solution of the descent search.
        fval: Function value at the solution of the search.
        gfval: Function gradient value at the solution of the search.
    """

    # Print header
    gradient_descent_print_header()

    # Initialize counter of iterations
    cnt_iter = 0
    # Initialize guess
    x = x0
    # Initialize stepsize
    step_size_k = alpha

    # Start
    while True:

        # Evaluate function
        fval_k = f(x)
        # Evaluate gradient
        gfval_k = gf(x)
        # Evaluate norm of gradient
        norm_gfval_k = np.linalg.norm(gfval_k)

        # Print 
        gradient_descent_print_iteration(cnt_iter, fval_k, norm_gfval_k, step_size_k)
        # Every 30 iterations print header
        if cnt_iter % 30 == 29:
            gradient_descent_print_header()

        # Check stopping conditions
        if gradient_descent_convergence_criterion(norm_gfval_k, eps) or gradient_descent_exceeded_iterations(cnt_iter, max_iter):
            break

        # Update solution
        x = x - step_size_k * gfval_k

        # Update iteration counter
        cnt_iter += 1
    
    # Print output message
    gradient_descent_print_output(cnt_iter, fval_k, norm_gfval_k, eps, max_iter)
    
    # Return
    return x, fval_k, gfval_k


if __name__ == "__main__":
    gradient_descent_print_header()
    gradient_descent_print_iteration(0, 12.3123, 0.23123, 1)