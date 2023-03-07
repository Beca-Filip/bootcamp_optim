import numpy as np

class LineSearch:
    def __init__(self, f, grad, hess = None, alpha = 0.5, alpha_max = 10, beta = 0.8, max_iter = 1e3, eps = 1e-6, ls_type = "backtracking", step_type = "l2_steepest", cond = "Armijo", armijo_const = 1e-4, wolfe_curvature_const = 0.8, lin_solver = np.linalg.solve):
        """Initialize line search object with the cost function and its gradient, along with numerical and categorical parameters.

        Args:
            f (function handle): Function handle of the minimization objective function.
            grad (function handle): Function handle of the gradient of the minimization objective function.
            hess (function handle, optional): Function handle of the hessian of the minimization objective function. Defaults to None.
            alpha (float, optional): Initial guess for step size. Defaults to 0.5.
            alpha_max (float, optional): Maximum step size. Defaults to 10.
            beta (float, optional): Default decrease on backtracking. Defaults to 0.8.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1e3.
            eps (float, optional): Tolerance for convergence. Defaults to 1e-6.
            ls_type (str, optional): Linesearch type. Defaults to "backtracking".
            step_type (str, optional): Type of step ("newton", "l2_steepest", "l1_steepest", ...). Defaults to "l2_steepest".
            cond (str, optional): Conditions to check at each iteration. Defaults to "Armijo".
            armijo_const (float, optional): Constant in the checking of Armijo condition. Defaults to 1e-4.
            wolfe_curvature_const (float, optional): Constant in the checking of the stong Wolfe curvature condition. Defaults to 0.8.
            lin_solver (function handle, optional): Solver for linear systems. Defaults to np.linalg.solve.
        """

        # Initialize the object
        self._f = f
        self._grad = grad
        self._hess = hess
        self._alpha = alpha
        self._alpha_max = alpha_max
        self._beta = beta
        self._max_iter = max_iter
        self._eps = eps
        self._ls_type = ls_type
        self._step_type = step_type
        self._cond = cond
        self._armijo_const = armijo_const
        self._wolfe_curvature_const = wolfe_curvature_const
        self._lin_solver = lin_solver

    
    def __call__(self, x0: np.ndarray):
        """
        Performs a line search optimization algorithm on the function f.
        
        Args:
        - x: current point
        
        Returns:
        x: Solution of the descent search.
        fval: Function value at the solution of the search.
        gradfval: Function gradient value at the solution of the search.
        """
        # Print header
        self._print_header()

        # Initialize guess
        self._xval_k = x0
        # Initialize iteration counter
        self._iter_cnter = 0
        # Initialize current stepsize
        self._alpha_k = self._alpha

        # Start
        while True:

            # Evaluate function
            self._fval_k = self._f(self._xval_k)
            # Evaluate gradient
            self._gradval_k = self._grad(self._xval_k)
            # Evaluate hessian if step_type = newton
            if self._step_type == "newton":
                self._hessval_k = self._hess(self._xval_k)
            # Update hessian if step_type = "bfgs"
            elif self._step_type == "bfgs":
                self._bfgs_update()
            # Else maintain a None pointer to the 
            else:
                self._hessval_k = None 

            # Evaluate norm of gradient
            self._norm_gradfval_k = np.linalg.norm(self._gradval_k)
            
            # Print current iterate
            self._print_iteration()
            # Every 30 iterations print header
            if self._iter_cnter % 30 == 29:
                self._print_header()

            # Check stopping conditions
            if self._convergence_condition() or self._exceeded_maximum_iterations():
                break

            # Compute search direction
            self._search_dir_k = self._compute_search_direction()

            # Compute directional derivative
            self._dir_deriv_k = self._compute_current_directional_derivative()

            # Choose stepsize
            self._alpha_k = self._compute_stepsize()
                
            # Update solution
            self._xval_k = self._xval_k + self._alpha_k * self._search_dir_k

            # Update iteration counter
            self._iter_cnter += 1

        # Print output message
        self._print_output()

        # Return
        return self._xval_k, self._fval_k, self._gradval_k

    
    def _exceeded_maximum_iterations(self):
        """Determines if the algorithm exceeded maximum iteration count.

        Properties used:
            self._iter_cnter (int): Iteration counter.
            self._max_iter (int): Maximum number of iterations.

        Returns:
            bool: Did the iteration counter exceed the maximum number of iterations?
        """
        return self._iter_cnter > self._max_iter
    
    
    def _convergence_condition(self):
        """Determines if the algorithm attained the desired tolerance in satisfying the convergence conditions.

        Properties used:
            self._norm_gradfval_k (float): Norm of the gradient.
            self._eps (float): Convergence tolerance.

        Returns:
            bool: Is the norm of the gradient smaller than the tolerance?
        """
        return self._norm_gradfval_k < self._eps

    
    def _print_header(self):
        """Prints the header of the optimization textual output.
        """
        print("{:<7}".format("Iter.") + " | " + "{:^10}".format("f(x)") +
            " | " + "{:^10}".format("||df(x)||") + " | " + "{:^10}".format("||d||"))
    

    def _print_iteration(self):
        """Prints the information of a single iteration of the gradient descent algorithm.
        
        Properties used:
            self._iter_cnter (int): Iteration counter.
            self._fval_k (float): Function value at current iterate.
            self._norm_gradfval_k (float): Gradient norm value at current iterate.
            self._alpha_k (float): Size of the step at current iterate.
        """
        print(("%7d" % self._iter_cnter) + " | " + ("%.4e" % self._fval_k) + " | " +
            ("%.4e" % self._norm_gradfval_k) + " | " + ("%.4e" % self._alpha_k))


    def _print_output(self):
        """Prints the final message of the search.
        
        Properties used:
            self._iter_cnter (int): Iteration counter.
            self._fval_k (float): Function value at current iterate.
            self._norm_gradfval_k (float): Gradient norm value at current iterate.
        """
        # If the algorithm converged
        if self._convergence_condition():
            print()
            print("Gradient descent successfully converged in %d iterations." % self._iter_cnter)
            print("Optimal function value: %.4e." % self._fval_k)
            print("Optimality conditions : %.4e." % self._norm_gradfval_k)

        # If the algorithm exceeded the iterations
        if self._exceeded_maximum_iterations():
            print()
            print("Gradient descent exceeded the maximum number (%d) of iterations." % self._max_iter)
            print("Current function value: %.4e." % self._fval_k)
            print("Current optimality conditions : %.4e." % self._norm_gradfval_k)

    
    def _compute_current_directional_derivative(self):
        """Computes the directional derivative at current iterate.
        Returns:
            float: Directional derivative at current iterate.
        """
        # Product between current gradient and current search direction
        return np.dot(self._gradval_k, self._search_dir_k)
    
    
    def _compute_directional_derivative(self, xval : np.ndarray, search_dir : np.ndarray):
        """Computes the directional derivative at a given point, for a given direction.
        Args:
            xval (np.ndarray): Point at which to calculate the directional derivative.
            search_dir (np.ndarray): Direction along which to calculate the directional derivative.
        Returns:
            float: Directional derivative at current iterate.
        """
        # Compute gradient at the point
        gradval = self._grad(xval)
        # Return directional derivative as the dot product between the gradient
        # and the search direction
        return np.dot(gradval, search_dir)
    
    
    def _compute_search_direction(self):
        """Computes the search direction for the line search.

        Properties used:
            self._step_type (string): Type of search direction to compute.
            self._gradval_k (np.ndarray): Function gradient value at current iterate.
            self._hessval_k (np.ndarray, optional): Value of the hessian fonction at the current iteration. Defaults to None.

            self._lin_solver (function handle, optional): Solver for linear systems. Defaults to np.linalg.solve.
        """
        # If we want the steepest descent direction in the L2 norm
        if self._step_type == "l2_steepest":
            return -self._gradval_k
        if self._step_type == "newton":
            return self._lin_solver(self._hessval_k, -self._gradval_k)
        if self._step_type == "bfgs":
            return self._compute_bfgs_step()
    
    
    def _armijo_condition_is_true(self, alpha, fval_alpha):
        """Check whether the Armijo condition is respected for a given step size, and function value corresponding to the step size. 
        Args:
            alpha (float): Value of step-size for which to check the condition.
            fval_alpha (float): Value of function at the point given by the stepsize.
        Properties used:
            self._f (function handle): Function handle of the minimization objective function.
            self._xval_k (np.ndarray): Value of the current iterate.
            self._search_dir_k (np.ndarray): The search direction at current iterate.
            self._dir_deriv_k (float): The directional derivative of the function along current search direction.        
        Returns
            bool: True if Armijo condition is respected.
        """
        # Return truth value of Armijo condition
        return fval_alpha <= self._fval_k + self._armijo_const * alpha * self._dir_deriv_k
    
    
    def _strong_wolfe_curvature_condition_is_true(self, dir_deriv_alpha):
        """Check whether the strong Wolfe curvature condition is respected for a given directional derivative value.

        Args:
            dir_deriv_alpha (float): Directional derivative value.
        Returns:
            bool: True if strong Wolfe curvature condition is respected.
        """
        # Return the truth value of the strong Wolfe curvature condition
        return np.abs(dir_deriv_alpha) <= - self._wolfe_curvature_const * self._dir_deriv_k
    
    
    def _compute_stepsize(self):
        """Chooses the stepsize of the line search.

        Args:
            ls_type (str, optional): _description_. Defaults to "backtracking".
        """
        # When the type is backtracking
        if self._ls_type == "backtracking":
            # Call the backtracking method
            return self._backtracking()
        # When the type is advanced
        if self._ls_type == "advanced":
            # Call the advanced line search algorithm
            return self._advanced_line_search()
            
    
    def _backtracking(self):
        """Calculates a step using backtracking.

        Returns:
            float: Step value computed by the backtracking.
        """        
        # Initialize the step iterate
        alpha = self._alpha        
        # Repeat
        while True:
            # Caclulate current function value
            fval_curr = self._f(self._xval_k + alpha * self._search_dir_k)
            # Check stopping conditions
            if self._armijo_condition_is_true(alpha=alpha, fval_alpha=fval_curr):
                break
            # Otherwise diminish alpha
            alpha = self._beta * alpha        
        # Return
        return alpha

    
    def _advanced_line_search(self):
        """Computes the step using an advanced line search algorithm (see Algorithm 3.2.,  Nocedal, Wright (2000), pp. 59).
        """
        # Initialize previous step iterate to 0
        alpha_prev = 0
        # Initialize previous function value
        fval_prev = self._fval_k
        # Initialize previous directional derivative
        dir_deriv_prev = self._dir_deriv_k
        # Initialize current step iterate to alpha
        alpha_curr = self._alpha
        # Line search iteration counter
        ls_iter_cnter = 1
        # Repeat
        while True:
            # Calculate current trial value
            xval_curr = self._xval_k + alpha_curr * self._search_dir_k
            # Caclulate current function value
            fval_curr = self._f(xval_curr)

            # Check negation of Armijo condition or increase condition
            if not self._armijo_condition_is_true(alpha=alpha_curr, fval_alpha=fval_curr) or (ls_iter_cnter > 1 and fval_curr >= fval_prev):
                # When the Armijo condition is violated, or the increase condition is true
                # Find a smaller step size satisfying the Armijo condition using _zoom
                return self._zoom(alpha_prev, alpha_curr, fval_prev, fval_curr, dir_deriv_prev, self._compute_directional_derivative(xval_curr, self._search_dir_k))
            
            # Calculate current directional derivative
            dir_deriv_curr = self._compute_directional_derivative(xval=xval_curr, search_dir=self._search_dir_k)
            # Check strong Wolfe curvature condition for the computed directional derivative
            if self._strong_wolfe_curvature_condition_is_true(dir_deriv_alpha=dir_deriv_curr):
                # At this point the Armijo and the strong Wolfe curvature conditions are satisfied
                # Return the current step
                return alpha_curr
            
            # At this point, the Armijo condition is satisfied, but the strong Wolfe conditions are not
            # If the directional derivative is positive here, it means the function keeps increasing so we stop
            # increasing the step size and return a smaller step size satisfying the Armijo and Wolfe conditions
            # using _zoom
            if dir_deriv_curr >= 0:
                return self._zoom(alpha_curr, alpha_prev, fval_curr, fval_prev, dir_deriv_curr, dir_deriv_prev)
            
            # At this point, the Armijo condition is satisfied, the strong Wolfe conditions are not, 
            # meaning the function is still decrasing strongly, indicating we may take a bigger step
            # If the step is already at maximum value return the current step
            if alpha_curr >= self._alpha_max:
                return self._alpha_max
            # Otherwise 
            else:
                # Update the previous step
                alpha_prev = alpha_curr
                fval_prev = fval_curr
                dir_deriv_prev = dir_deriv_curr
                # Increase the current step by a factor of 1/self._beta
                alpha_curr = min([alpha_curr / self._beta, self._alpha_max])
            
            # Update iteration counter
            ls_iter_cnter += 1

    
    def _zoom(self, alpha_lo : float, alpha_hi : float, fval_lo : float, fval_hi : float, dir_deriv_lo : float, dir_deriv_hi : float):
        """Performs the zoom algorithm for line searching a step length satisfying the strong wolfe conditions (Algorithm 3.3., Nocedal, Wright (2000), pp. 60).

        Args:
            alpha_lo (float): Step size with the lowest function value currently found.
            alpha_hi (float): Considered step size with the property dir_deriv_lo * (alpha_hi - alpha_lo) < 0.
            fval_lo (float): Function value at iterate with step size alpha_lo.
            fval_hi (float): Function value at iterate with step size alpha_hi.
            dir_deriv_lo (float): Directional derivative at iterate with step size alpha_lo.
            dir_deriv_hi (float): Directional derivative at iterate with step size alpha_hi.
            
        Returns:
            float: Step size satisfying the strong Wolfe conditions.
        """
        # Repeat
        while True:
            # Find a next step size to consider using quadratic or cubic interpolation
            alpha_curr = self._cubic_step_interpolation(alpha_lo, alpha_hi, fval_lo, dir_deriv_lo, fval_hi, dir_deriv_hi)

            # Evaluate current point
            xval_curr = self._xval_k + alpha_curr * self._search_dir_k
            # Evaluate current function value
            fval_curr = self._f(xval_curr)

            # Check negation of Armijo condition or increase condition w.r.t. fval_lo
            if not self._armijo_condition_is_true(alpha=alpha_curr, fval_alpha=fval_curr) or fval_curr >= fval_lo:
                # When the Armijo condition is violated, or the increase condition is true
                # Update alpha_hi and its information
                alpha_hi = alpha_curr
                # Update function value
                fval_hi = fval_curr
                # Update directional derivative
                dir_deriv_hi = self._compute_directional_derivative(xval_curr, self._search_dir_k)

            # Otherwise, if Armijo condition is true
            else:
                # Compute the current directional derivative
                dir_deriv_curr = self._compute_directional_derivative(xval=xval_curr, search_dir=self._search_dir_k)

                # Check strong Wolfe curvature condition for the computed directional derivative
                if self._strong_wolfe_curvature_condition_is_true(dir_deriv_alpha=dir_deriv_curr):
                    # At this point the Armijo and the strong Wolfe curvature conditions are satisfied
                    # Return the current step
                    return alpha_curr
                
                # At this point the Armijo condition is true and the strong Wolfe curvature condition is false
                # Meaning we found a better point than alpha_lo in terms of function value but still haven't found
                # a point satisfying the strong Wolfe curvature condition. So we need to set alpha_lo to alpha_curr.

                # However, if the directional derivative at alpha_curr is a different sign than the directional derivative at alpha_lo
                # we need to update alpha_hi
                if dir_deriv_curr * (alpha_hi - alpha_lo) >= 0:
                    # Replace alpha_hi by alpha_lo
                    alpha_hi = alpha_lo
                    # Replace fval_hi by fval_lo
                    fval_hi = fval_lo
                    # Replace directional derivative
                    dir_deriv_hi = dir_deriv_lo

                # Update step size
                alpha_lo = alpha_curr
                # Update function value
                fval_lo = fval_curr
                # Update directional derivative
                dir_deriv_lo = dir_deriv_curr


    def _quadratic_step_interpolation(self, alpha_0 : float, alpha_1 : float, phi_0 : float, phi_prim_0 : float, phi_1 : float):
        """Finds the step that is the minimum of the quadratic interpolation of the line-function. Defaults to bisection
        if step is outside bounds defined by alphas.

        Args:
            alpha_0 (float): Step size where the function and the directional derivative value are used.
            alpha_1 (float): Step size where only the function value is used.
            phi_0 (float): Function value at step alpha0.
            phi_prim_0 (float): Directional derivative value at step alpha0.
            phi_1 (float): Function value at step alpha1.
        """
        # Intermediate calculations
        diff_phi = phi_1 - phi_0
        diff_alpha = alpha_1 - alpha_0
        sum_alpha = alpha_1 + alpha_0
        diff_sq_alpha = diff_alpha * sum_alpha
        # Calculate minimal alpha
        alpha_star = (-phi_prim_0 * diff_sq_alpha + 2 * alpha_0 * diff_phi) / (diff_phi - diff_alpha * phi_prim_0) / 2
        # Intermediate calculations
        min_alpha = min([alpha_0, alpha_1])
        max_alpha = max([alpha_0, alpha_1])
        # Safeguard
        if alpha_star <= min_alpha or alpha_star >= max_alpha:
            alpha_star = sum_alpha / 2
        # Return
        return alpha_star
    

    def _cubic_step_interpolation(self, alpha_0 : float, alpha_1 : float, phi_0 : float, phi_prim_0 : float, phi_1 : float, phi_prim_1 : float):
        """Finds the step that is the minimum of the cubic interpolation of the line-function. Defaults to bisection
        if step is outside bounds defined by alphas.

        Args:
            alpha_0 (float): Step size where the function and the directional derivative value are used.
            alpha_1 (float): Step size where only the function value is used.
            phi_0 (float): Function value at step alpha0.
            phi_prim_0 (float): Directional derivative value at step alpha_0.
            phi_1 (float): Function value at step alpha1.
            phi_prim_1 (float): Directional derivative value at step alpha_1.
        """
        # Intermediate computations
        d1 = phi_prim_0 + phi_prim_1 - 3 * (phi_0 - phi_1) / (alpha_0 - alpha_1)
        d2 = np.sqrt(d1**2 - phi_prim_0 * phi_prim_1)
        # Compute optimal alpha
        alpha_star = alpha_1 - (alpha_1 - alpha_0) * (phi_prim_1 + d2 - d1) / (phi_prim_1 - phi_prim_0 + 2 * d2)
        # Intermediate calculations
        min_alpha = min([alpha_0, alpha_1])
        max_alpha = max([alpha_0, alpha_1])
        # Safeguard
        if alpha_star <= min_alpha or alpha_star >= max_alpha:
            alpha_star = (alpha_0 + alpha_1) / 2
        return alpha_star

if __name__ == "__main__":

    X = []