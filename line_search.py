import numpy as np

def line_search_exceeded_iterations(cnt_iter : int, max_iter : int):
    """Determines if the algorithm exceeded maximum iteration count.

    Args:
        cnt_iter (int): Iteration counter.
        max_iter (int): Maximum number of iterations.

    Returns:
        bool: Did the iteration counter exceed the maximum number of iterations?
    """
    return cnt_iter > max_iter

def line_search_convergence_criterion(norm_gradfval_k : float, eps : float):
    """Determines if the algorithm attained the desired tolerance in satisfying the convergence conditions.

    Args:
        norm_gradfval_k (float): Norm of the gradient.
        eps (float): Epsilon tolerance.

    Returns:
        bool: Is the norm of the gradient smaller than the tolerance?
    """
    return norm_gradfval_k < eps

def line_search_print_header():
    """Prints the header of the optimization textual output.
    """
    print("{:<7}".format("Iter.") + " | " + "{:^10}".format("f(x)") + " | " + "{:^10}".format("||df(x)||") + " | " + "{:^10}".format("||d||"))

def line_search_print_iteration(cnt_iter : int, fval_k : float, norm_gradfval_k : float, step_size_k : float):
    """Prints the information of a single iteration of the gradient descent algorithm.

    Args:
        cnt_iter (int): Iteration counter.
        fval_k (float): Function value at current iterate.
        norm_gradfval_k (float): Gradient norm value at current iterate.
        step_size_k (float): Size of the step at current iterate.
    """
    print(("%7d" % cnt_iter) + " | " + ("%.4e" % fval_k) + " | " + ("%.4e" % norm_gradfval_k) + " | " + ("%.4e" % step_size_k))

def line_search_print_output(cnt_iter : int, fval_k : float, norm_gradfval_k : float, eps : float, max_iter : int):
    """Prints the final message of the search.

    Args:
        cnt_iter (int): Iteration counter.
        fval_k (float): Function value at current iterate.
        norm_gradfval_k (float): Gradient norm value at current iterate.
        eps (float): Tolerance on the first order optimality condition for convergence.
        max_iter (int): Maximum number of iterations to perform.
    """
    # If the algorithm converged
    if line_search_convergence_criterion(norm_gradfval_k=norm_gradfval_k, eps=eps):
        print()
        print("Gradient descent successfully converged in %d iterations." % cnt_iter)
        print("Optimal function value: %.4e." % fval_k)
        print("Optimality conditions : %.4e." % norm_gradfval_k)

    # If the algorithm exceeded the iterations
    if line_search_exceeded_iterations(cnt_iter=cnt_iter, max_iter=max_iter):
        print()
        print("Gradient descent exceeded the maximum number (%d) of iterations." % max_iter)
        print("Current function value: %.4e." % fval_k)
        print("Current optimality conditions : %.4e." % norm_gradfval_k)

def line_search_compute_search_direction(fval_k : float, gradfval_k : np.ndarray, step_type="l2_steepest", hessf=[], ns_solver=np.linalg.solve):
    """Computes the search direction for the line search.

    Args:
        fval_k (float): Function value at current iterate.
        gradfval_k (np.ndarray): Function gradient value at current iterate.
        step_type (str, optional): Type of step ("newton", "l2_steepest", "l1_steepest", ...). Defaults to "steepest".
        hessf (list, optional): _description_. Defaults to [].
        ns_solver (_type_, optional): _description_. Defaults to np.linalg.solve.
    """
    # If we want the steepest descent direction in the L2 norm
    if step_type == "l2_steepest":
        return -gradfval_k
    
def line_search_armijo_condition_is_true(f, fval_k, gradfval_k, x, alpha, search_dir_k):
    """_summary_

    Args:
        f (_type_): _description_
        fval_k (_type_): _description_
        gradfval_k (_type_): _description_
        x (_type_): _description_
        alpha (_type_): _description_
        search_dir_k (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Armijo number
    c = 0.8
    # Calculate the value of the function at the next iter
    fnext = f(x + alpha * search_dir_k)
    # Calculate the directional derivative along the search direction
    dirderiv = alpha * np.dot(gradfval_k, search_dir_k)

    # Return Armijo condition
    return fnext < fval_k + c * dirderiv

    
def line_search_choose_stepsize(f, gradf, fval_k : float, gradfval_k : np.ndarray, x : np.ndarray, search_dir_k : np.ndarray, alpha : float, ls_type="backtracking"):
    """Chooses the stepsize of the line search.

    Args:
        f (_type_): _description_
        gf (_type_): _description_
        fval_k (float): _description_
        gradfval_k (np.ndarray): _description_
        x (np.ndarray): _description_
        search_dir_k (np.ndarray): _description_
        alpha (float): _description_
        ls_type (str, optional): _description_. Defaults to "backtracking".
    """
    # When the type is backtracking
    if ls_type == "backtracking":

        # Choose ro
        ro = 0.7
        # Initialize the step iterate
        alpha = alpha
        
        # Repeat
        while True:

            # Check stopping conditions
            if line_search_armijo_condition_is_true(f, fval_k, gradfval_k, x, alpha, search_dir_k):
                break

            # Otherwise diminish alpha
            alpha = ro * alpha
        
        # Return
        return alpha
        
    

def line_search(f, gradf, alpha : float, x0 : np.ndarray, eps=1e-4, max_iter=1e3, ls_type="backtracking", step_type="l2_steepest", hessf = [], ns_solver=np.linalg.solve):
    """ Performs a line search optimization algorithm on the function f.

    Args:
        f (function): Function handle of the function to minimize.
        gradf (function): Function handle of the gradient of the function to minimize.
        alpha (float): Maximum step size.
        x0 (np.ndarray): Initial guess for the algorithm.
        eps (float, optional): Tolerance on the first order optimality condition for convergence. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations to perform. Defaults to 1e3.
        ls_type (str, optional): Type of line search. Defaults to "backtracking".
        step_type (str, optional): Type of step ("newton", "l2_steepest", "l1_steepest", ...). Defaults to "steepest".
        hessf (function, optional): Function handle of the hessian of the function to minimize. Defaults to [].
        ns_solver (function, optional): Function handle of the solver to use for solving the Newton step. Defaults to np.linalg.solve.


    Returns:
        x: Solution of the descent search.
        fval: Function value at the solution of the search.
        gradfval: Function gradient value at the solution of the search.
    """

    # Print header
    line_search_print_header()

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
        gradfval_k = gradf(x)
        # Evaluate norm of gradient
        norm_gradfval_k = np.linalg.norm(gradfval_k)

        # Print 
        line_search_print_iteration(cnt_iter, fval_k, norm_gradfval_k, step_size_k)
        # Every 30 iterations print header
        if cnt_iter % 30 == 29:
            line_search_print_header()

        # Check stopping conditions
        if line_search_convergence_criterion(norm_gradfval_k, eps) or line_search_exceeded_iterations(cnt_iter, max_iter):
            break

        # Compute search direction
        search_dir_k = line_search_compute_search_direction(fval_k, gradfval_k, step_type, hessf, ns_solver)

        # Choose stepsize
        step_size_k = line_search_choose_stepsize(f, gradf, fval_k, gradfval_k, x, search_dir_k, alpha, ls_type)

        # Update solution
        x = x + step_size_k * search_dir_k

        # Update iteration counter
        cnt_iter += 1
    
    # Print output message
    line_search_print_output(cnt_iter, fval_k, norm_gradfval_k, eps, max_iter)
    
    # Return
    return x, fval_k, gradfval_k


if __name__ == "__main__":
    line_search_print_header()
    line_search_print_iteration(0, 12.3123, 0.23123, 1)
    print(line_search_compute_search_direction(0.3, np.array([0.3, -0.3])))
    print(line_search_choose_stepsize)