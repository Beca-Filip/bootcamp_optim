import numpy as np

def gradient_descent_exceeded_iterations(cnt_iter, max_iter):
    return cnt_iter > max_iter

def gradient_descent_convergence_criterion(norm_gfval_k, eps):
    return norm_gfval_k < eps

def gradient_descent_print_header():
    print("{:^7}".format("Iter.") + " | " + "{:^10}".format("f(x)") + " | " + "{:^10}".format("||df(x)||") + " | " + "{:^10}".format("||d||"))

def gradient_descent_print_iteration(cnt_iter, fval_k, norm_gfval_k, step_size_k):
    print(("%7d" % cnt_iter) + " | " + ("%.4e" % fval_k) + " | " + ("%.4e" % norm_gfval_k) + " | " + ("%.4e" % step_size_k))


def gradient_descent(f, gf, alpha, x0, eps=1e-4, max_iter=1e3):

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

        # Check stopping conditions
        if gradient_descent_convergence_criterion(norm_gfval_k, eps) or gradient_descent_exceeded_iterations(cnt_iter, max_iter):
            break

        # Update solution
        x = x - step_size_k * gfval_k

        # Update iteration counter
        cnt_iter += 1
    
    # Print output message
    # gradient_descent_print_output()


if __name__ == "__main__":
    gradient_descent_print_header()
    gradient_descent_print_iteration(0, 12.3123, 0.23123, 1)