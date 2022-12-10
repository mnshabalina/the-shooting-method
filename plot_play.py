"""
Install dependencie first with
    pip install -r reuirements.txt
"""
import scipy
import numpy as np
import matplotlib.animation as animation
from math import pi, cos, sin, exp
from matplotlib import pyplot as plt

from shooting import Shooter, ode_k, y_k, t1, t0


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
        color = self.colors[i]

        # plt.figure(figsize=(10, 10))
        # plt.title(f'Error')
        # plt.xlabel('t')
        # plt.scatter(self.t, errors, s=5, color=color, label=f'k={k}')
        # plt.plot(self.t, errors, linestyle='--', color='red', linewidth=1, label='error shooting')
                
        # plt.figure(figsize=(5, 5))
        # xs = np.arange(len(self.guess_history)) + 1
        # ys = self.true_boundary - np.array(self.boundary_history)
        # plt.axes().set_xticks(xs)
        # plt.axhline(0, linestyle='--', color='gray', linewidth=1)
        # plt.title(f'Boundary error')
        # plt.xlabel('Iteration number')
        # plt.scatter(xs, ys, s=10)
        # plt.plot(xs, ys, linestyle='--', color='blue', linewidth=1)

        # plt.figure(figsize=(10, 10))
        plt.title(f'Exact solution vs approximation')
        plt.scatter(self.t, self.solution, color='red', s=5, label=f'Shooting')
        plt.plot(self.t, y_true, color='green', linewidth=1, label=f'y true')
        plt.legend()
        plt.show()


        # Compare with the numeric solution of the original BVP
        
        y_bvp = self._solve_bvp()
        # errors_bvp = self._error_report(y_true, y_bvp, name='BVP')

        # plt.figure(figsize=(10, 10))
        # plt.title(f'Error BVP')
        # plt.xlabel('t')
        # plt.scatter(self.t, errors_bvp, s=5linestyle='--', color='blue',)
        # plt.plot(self.t, errors_bvp, linestyle='--', color='blue', linewidth=1, label='error BVP')


        # plt.figure(figsize=(10, 10))
        # plt.title(f'Exact solution vs BVP approximation')
        # plt.scatter(self.t, y_bvp, color=color, s=5)#, label=f'BVP k={k}')
        # plt.plot(self.t, y_true, color='green', linewidth=1, label='true y')
        # plt.legend()
        
        # plt.figure(figsize=(10, 10))
        # plt.title(f'Exact solution vs BVP numeric solution vs Shooting')
        # plt.scatter(self.t, y_bvp, color='blue', s=7, label='BVP')
        # plt.plot(self.t, y_true, color='green', linewidth=1, label='true y')
        # plt.scatter(self.t, self.solution, color='red', s=10, label='Shooting')
        # plt.legend()

        # plt.figure(figsize=(10, 10))
        # plt.title(f'Error Shooting vs BVP')
        plt.xlabel('t')
        # plt.scatter(self.t, errors, s=15, color='blue',)
        # plt.plot(self.t, errors, linestyle='--', color='blue', linewidth=1, label='Shooting')
        # plt.scatter(self.t, errors_bvp, s=15, color='green',)
        # plt.plot(self.t, errors_bvp, linestyle='--', color='green', linewidth=1, label='BVP')
        # plt.legend()

        # plt.show()



if __name__ == "__main__":

   
    # color = 'blue'
    plt.figure(figsize=(10,10))
    # for k in [5, 10, 15, 20, 25]:
    #     t = np.linspace(t0, t1, 1000)
    #     def y1(x):
    #         return (cos(pi*x)             
    #             + (
    #                 + exp(k*(x-1))         
    #                 + exp(2*k*(x-1))       
    #                 + exp(-k*x)            
    #             ) / (2+exp(-k)))    
        
    #     plt.plot(t, [y1(x) for x in t], label=f'k={k}')
        
    #     # plt.plot([self.t_span[1]], [boundary], marker="x", markersize=7, markerfacecolor=color)
    #     # plt.axvline(self.t_span[1], linestyle='--', color='lightgray', linewidth=1)
    # plt.legend()
    # plt.show()
    i=-1
    plt.title('Error Shooting')
    for k in [5, 10, 15]:
        i+=1
        print(k)
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

    plt.legend()

    plt.show()
        # try:
        #     shooter.animate_history()
        # except Exception as e:
        #     print("Animation failed. Try checking ffmpeg.")
        #     print("Error message:")
        #     print(e)
        # print('------- Solution -------')
        # print(solution)
