"""
Install dependencie first with
    pip install -r reuirements.txt
"""
import scipy
import numpy as np
import matplotlib.animation as animation
from math import pi, cos, sin, exp
from matplotlib import pyplot as plt


class Shooter:
    """
    Implements the shooting method.

    optimize() solves the intialized BVP with the shooting method
    animate_history() animates the shooting history (ffmpeg is required)
    evaluate_results(y_true) prints and plots error of the approximation
    """

    def __init__(self, bvp, n_steps=100):
        """
        Initialize the solver.

        Args:
            bvp (dict): the boundary value problem in the form:
                {
                    'ode' (callable): function of Y that returns Y': 
                            ode([y1, y2, ... yn]) = [y'1, y'2, ..., y'n],
                    't_span' (array-kike): [t0, t1] interval to solve the BVP on,
                    'Y0' (array-kike): [y1(t0), y2(t0), ... yn-1(t0)],
                    'Y1' (float): y1(t1)
                }
            n_steps (int, optional): the number of points to compute the solution at. Defaults to 100.
        """
        self.ode = bvp['ode']
        self.Y0 = bvp['Y0']
        self.true_boundary = bvp['Y1']
        self.t_span = bvp['t_span']
        self.n_steps = n_steps

        self.t = np.linspace(self.t_span[0], self.t_span[1], n_steps)

        self.solution_history = []
        self.guess_history = []
        self.boundary_history = []

        self.colors = ['blue', 'orange', 'purple', 'green', 'black']

    def optimize(self, initial_guess=0):
        # Find the zero of the error function, which is the right initial value
        self.root, output = scipy.optimize.newton(self._error, initial_guess, full_output=True)
        print('\n------- Newton optimizer report -------\n')
        print(output)
        print(f'\nBoundary error: {self._error(self.root)}')
        # The last solution in history is the right one
        self.solution = self.solution_history[-1].T[:,0]

        return self.solution

    def animate_history(self):
        plt.figure(figsize=(10, 10))
        shoot_animation = animation.FuncAnimation(
                                        plt.gcf(),
                                        self._animate,
                                        frames=range(len(self.guess_history)),
                                        interval=1000,
                                        repeat=False)
        shoot_animation.save(f'history_{self.true_boundary}.mp4', writer='ffmpeg')
        plt.show()
        
    def evaluate_results(self, y_true):
        """
        Print and plot error of the approximation.

        Args:
            y_true (array-kike of callables): [y1(t), y2(t), ... yn(t)]
        """
        # Check that y_true ist indeed the exact solution of the BVP.
        print('\n------- Check the exact solution before evaluation -------\n')
        passed = self._check_exact_solution(y_true)
        if not passed:
            print('The provided y is not the exact solution of the BVP!\n')
            return
        print('Check passed.\n')
        
        # Evaluate the true y on the interval
        y_true = [y_true[0](x) for x in self.t]
        
        errors = self._error_report(y_true, self.solution)

        plt.figure(figsize=(10, 10))
        plt.title(f'Error')
        plt.xlabel('t')
        plt.scatter(self.t, errors, s=15)
        plt.plot(self.t, errors, linestyle='--', color='blue', linewidth=1)
                
        plt.figure(figsize=(5, 5))
        xs = np.arange(len(self.guess_history)) + 1
        ys = self.true_boundary - np.array(self.boundary_history)
        plt.axes().set_xticks(xs)
        plt.axhline(0, linestyle='--', color='gray', linewidth=1)
        plt.title(f'Boundary error')
        plt.xlabel('Iteration number')
        plt.scatter(xs, ys, s=10)
        plt.plot(xs, ys, linestyle='--', color='blue', linewidth=1)

        plt.figure(figsize=(10, 10))
        plt.title(f'Exact solution vs approximation')
        plt.scatter(self.t, self.solution, color='red', s=10, label='approximation')
        plt.plot(self.t, y_true, color='green', linewidth=1, label='true y')
        plt.legend()


        # Compare with the numeric solution of the original BVP
        
        y_bvp = self._solve_bvp()
        errors_bvp = self._error_report(y_true, y_bvp, name='BVP')

        plt.figure(figsize=(10, 10))
        plt.title(f'Error BVP')
        plt.xlabel('t')
        plt.scatter(self.t, errors_bvp, s=15)
        plt.plot(self.t, errors_bvp, linestyle='--', color='blue', linewidth=1)


        plt.figure(figsize=(10, 10))
        plt.title(f'Exact solution vs BVP approximation')
        plt.scatter(self.t, y_bvp, color='red', s=10, label='BVP approximation')
        plt.plot(self.t, y_true, color='green', linewidth=1, label='true y')
        plt.legend()
        
        plt.figure(figsize=(10, 10))
        plt.title(f'Exact solution vs BVP numeric solution vs Shooting')
        plt.scatter(self.t, y_bvp, color='blue', s=10, label='BVP numeric solution')
        plt.plot(self.t, y_true, color='green', linewidth=1, label='true y')
        plt.scatter(self.t, self.solution, color='red', s=10, label='Shooting')
        plt.legend()

        plt.figure(figsize=(10, 10))
        plt.title(f'Error Shooting vs BVP')
        plt.xlabel('t')
        plt.scatter(self.t, errors, s=15, color='blue',)
        plt.plot(self.t, errors, linestyle='--', color='blue', linewidth=1, label='Shooting')
        plt.scatter(self.t, errors_bvp, s=15, color='green',)
        plt.plot(self.t, errors_bvp, linestyle='--', color='green', linewidth=1, label='BVP')
        plt.legend()

        plt.show()

    def _error_report(self, y_true, approximation, name='Shooting'):
        
        approximation = np.array(approximation)
        y_true = np.array(y_true)

        errors = y_true - approximation
        abs_errors = np.absolute(errors)
        max_abs_error = np.max(abs_errors)
        mean_abs_error = np.mean(abs_errors)

        print(f'\n------- Error report {name} -------\n')
        print(f'Max absolute error: {max_abs_error}')
        print(f'Mean absolute error: {mean_abs_error}')

        return errors


    def _shoot(self, guess):
        # Initial values
        y0 = self.Y0 + [guess]
        # Solve IVP
        self.ivp_solver = scipy.integrate.solve_ivp(self.ode,
                                                    self.t_span,
                                                    y0,
                                                    dense_output=True)
        # Evaluate solution on the interval
        self.ivp_solution = self.ivp_solver.sol(self.t)
        # The resulting boundary value 
        boundary = self.ivp_solution[0,-1]
        
        # Save intermidiate results for visualisation
        self.solution_history.append(self.ivp_solution)
        self.boundary_history.append(boundary)
        self.guess_history.append(guess)
        
        return boundary

    def _error(self, guess):
        # Compute boundary value by solving IVP with initial value = guess
        boundary = self._shoot(guess)
        # Difference between real boundary value and that computed based on the guess
        error = self.true_boundary - boundary
        return error

    def _check_exact_solution(self, y, tol=10**-8):
        max_abs_error = 0
        for t in self.t:
            y_t = [y_i(t) for y_i in y]
            yn_from_ode = self.ode(t, y_t[:-1])[-1]
            abs_error = abs(y_t[-1] - yn_from_ode)
            max_abs_error = max(abs_error, max_abs_error)
        print(f"Max absolute error of the exact solution: {max_abs_error}")
        return max_abs_error < tol

    def _solve_bvp(self):
        # Rearrange (vectorize) the data for the bvp solver
        def ode_bvp(t, y):
            # Compute for each t
            n = len(t)
            y_t = [self.ode(t[i], y[:,i]) for i in range(n)]
            # Transpose
            m = len(y_t[0])
            Y_T= [[y_t[j][i]for j in range(n)] for i in range(m)]
            return Y_T
        # The boundary conditions
        def bc(ya, yb):
            return np.array([ya[0]-self.Y0[0],
                             ya[1]-self.Y0[1],
                             yb[0]-self.true_boundary])
        # Initial approximation
        Y_0 = np.zeros((3, self.n_steps))
        # The solver is the 4th order collocation algorithm
        bvp_solver = scipy.integrate.solve_bvp(ode_bvp, bc, self.t, Y_0)
        y_bvp = bvp_solver.sol(self.t)[0]
        return y_bvp

    def _animate(self, i):
        self._plot_shoot(self.solution_history[i],  
                        self.boundary_history[i], 
                        i)

    def _plot_shoot(self, shoot_solution, boundary, i):
        color = self.colors[i % len(self.colors)]
        plt.title('Shooting history\nBoundary error at iteration '
                    f'{i}: {self.true_boundary - boundary}')
        plt.xlabel('t')
        plt.ylabel('y')
        if i==0:
            plt.plot([self.t_span[1]], [self.true_boundary], marker="o", markersize=10, markerfacecolor="red", color='red')
        plt.plot(self.t, shoot_solution[0,:], linestyle='--', color=color)
        plt.plot([self.t_span[1]], [boundary], marker="x", markersize=7, markerfacecolor=color)
        plt.axvline(self.t_span[1], linestyle='--', color='lightgray', linewidth=1)


if __name__ == "__main__":

    def ode_k(k):
        def q(x, k):
            return(
                    (k**2 + pi**2)
                    * (
                        + 2*k*cos(pi*x)
                        + pi*sin(pi*x)
                    )
                )

        def _ode(t, y):
            y1, y2, y3 = y
            return [y2, y3, -2*k**3*y1 + k**2*y2 + 2*k*y3 + q(t, k)]
        
        return _ode
        
    def y_k(k):
        def y1(x):
            return (cos(pi*x)             
                + (
                    + exp(k*(x-1))         
                    + exp(2*k*(x-1))       
                    + exp(-k*x)            
                ) / (2+exp(-k)))                                
        def y2(x):
            return (-pi*sin(pi*x) 
                + k*(
                    - exp(k-k*x)
                    + exp(k*x) 
                    + 2*exp(k*(2*x-1))
                ) / (1 + 2*exp(k)))

        def y3(x):
            return (-pi**2*cos(pi*x)
                + ( k**2*exp(-k*x)
                    * (
                        + exp(2*k*x)
                        + 4*exp(k*(3*x-1))
                        + exp(k)
                    )
                ) / (1 + 2*exp(k))) 

        def y4(x):
            return (pi**3*sin(pi*x)
                + ( k**3*exp(-k*x)
                    * (
                        + exp(2*k*x)
                        + 8*exp(k*(3*x-1))
                        - exp(k)
                    )
                ) / (1 + 2*exp(k))) 
        return [y1, y2, y3, y4]

    t0, t1 = 0, 1

    for k in [1, 10, 20, 100]:
        ode = ode_k(k)   
        y = y_k(k) 
        bvp = {
                'ode': ode,
                'Y0': [y[0](t0), y[1](t0)],
                'Y1': y[0](t1),
                't_span':[t0, t1],
        }

        kw_params = {
            'n_steps': 100,
            'bvp': bvp,
        }
    
        shooter = Shooter(**kw_params)
        solution = shooter.optimize()
        shooter.evaluate_results(y)

        # try:
        #     shooter.animate_history()
        # except Exception as e:
        #     print("Animation failed. Try checking ffmpeg.")
        #     print("Error message:")
        #     print(e)
        # print('------- Solution -------')
        # print(solution)
