import scipy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

class Shooter:

    def __init__(self, system, tol):
        self.tol = tol
        self.fun = system['fun']
        self.y1 = system['y1']
        self.y2 = system['y2']
        self.Y0 = system['Y0']
        self.t_span = system['t_span']
        self.t = np.linspace(self.t_span[0], self.t_span[1], 300)
        self.true_boundary = self.y1(self.t_span[1])
        self.solution_history = []
        self.guess_history = []
        self.boundary_history = []

    def error(self, guess):
        boundary = self.shoot(guess)
        error = self.true_boundary - boundary
        # print(f'boundary: {boundary}')
        # print(f'true_boundary: {self.true_boundary}')
        # print(f'error: {error}')
        return error

    def shoot(self, guess):
        y0 = (self.Y0, guess)
        self.ivp_solver = scipy.integrate.solve_ivp(self.fun, self.t_span, y0, dense_output=True, atol=self.tol)
        self.ivp_solution = self.ivp_solver.sol(self.t)
        boundary = self.ivp_solver.sol(self.t_span[1])[1]
        self.solution_history.append(self.ivp_solution)
        self.boundary_history.append(boundary)
        self.guess_history.append(guess)
        return boundary

    def plot_shoot(self, shoot_solution, guess, boundary, i=0):
        plt.plot(self.t, shoot_solution.T[:,0], linestyle='--')
        plt.plot(self.t, [self.y1(x) for x in self.t], color='green')
        plt.axvline(self.t_span[1], linestyle='--')
        plt.plot([self.t_span[1]], [self.true_boundary], marker="o", markersize=10, markerfacecolor="green")
        plt.plot([self.t_span[1]], [boundary], marker="o", markersize=10, markerfacecolor="red")
        plt.xlabel('t')
        plt.legend([f'y guessed iteration {i}', 'y true'], shadow=True)
        plt.title(f'guess ={guess}\ntrue = {self.y2(self.t_span[0])}\niter={i}')
        
    def animate(self, i):
        self.plot_shoot(self.solution_history[i], 
                        self.guess_history[i], 
                        self.boundary_history[i], 
                        i)
        
    def optimize(self, initial_guess=0):
        self.root = scipy.optimize.newton(self.error, initial_guess, maxiter=100, tol=self.tol)
        max_error = max( [ y_true - y_pred for y_true, y_pred in zip( [ self.y1(x) for x in self.t ], self.solution_history[-1].T[:,0] ) ] )
        print(f'Max error: {max_error}')
        print(f'N iterations to converge: {len(self.solution_history)}')
        return self

    def animate_history(self):
        plt.figure(figsize=(25, 50))
        shoot_animation = animation.FuncAnimation(plt.gcf(),
                                                  self.animate,
                                                  frames=range(len(self.guess_history)), 
                                                  interval=1000)
        plt.show()


if __name__ == "__main__":
    def fun(t, y):
        y1, y2 = y
        return [y2, -y2 + 2*y1 -2*t -1]

    system = {
            'fun': fun,
            'y1': lambda t: 5*np.exp(t)+t+1,
            'y2': lambda t: 5*np.exp(t)+1,
            'Y0': 6,
            't_span':[0, 5],
    }

    kw_params = {
        'tol': 0.00001,
        'system': system,
    }
    shooter = Shooter(**kw_params)
    shooter.optimize().animate_history()

