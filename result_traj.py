import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#Текущая аппроксимация в ILTT_Toolbox
def T_a(a, kappa, alt=True):
    """Время перелета из угловой дальности"""
    B = 0.2721831
    kappa = np.clip(kappa, 1e-6, np.inf)
    if alt:
        T = a*(0.03772064567906565*kappa + 1.405055577366462)*np.log(kappa + 1)+(0.08297934358955021*kappa + 0.009838514316752341)*np.log(kappa)**2
    else:
        T =  (kappa - kappa * B * np.log(kappa)) * a + (kappa * B**2 * np.log(kappa))
    if a < 0.75:
        K = kappa
        L = a
        T = (-L**(-K)*(0.7521363*K + 0.24717593)*(K*(0.0014925372**K*math.log(K + L) - 0.0003217671) - L**(K + 1)*(L + 0.0911687))/(L + 0.0911687)) #3.5e-03
    return T
def get_initial_adjoint(a, k, d=1):
    if a < 0.75:
        # --- MARS ---        
        def pr_x_M(L, d):
            y = (-(-0.5683159) + (0.75718486*math.exp(math.cos(L*(L + 1.8499389) + L + L + 0.75718486))**0.008454218 - math.sin(0.8540929 + 0.43165326/L))*(L*math.exp(L) + 0.06105759)/L) #1.7e-04
            return y
        def pr_y_M(L, d):
            y = ((-0.0193943978884244 + 1.3159028*math.cos(0.20418811467131*(L - 0.17080334)*math.log(L)/L)/(math.exp(5.2420635*L) - 0.38545114))*math.sin(0.4104828/L)) #1.1e-04
            return y
        def pv_x_M(L, d):
            y = (0.34306207*(1.5114436**L - (-0.0008686929)*math.exp(math.cos(-1.089295 - 1.5140266/L)))*(-L**(-1.3051801)*L*math.sin(L**(-0.7102793)) + (1.2651104*L)**(-0.7364218))) #7.0e-05
            return y
        def pv_y_M(L, d):
            y = ((math.sin(0.3880472/math.atan(L)) - 0.5640575)/((0.1749562 - math.atan(L))*(math.atan(math.sin(0.36976972/L)) - 1.3752396) + math.atan(L) + 1.2624102) + 0.05609478) #1.2e-04
            return y
        # --- VENUS ---            
        def pr_x_V(L, d):
            y = (-(-0.0015014085)*math.sin(-0.37150827 - 1.6895345/L) - 0.0036338682 + math.atan(math.sin((-0.03401584 - 0.1006823/L)/((L/0.19236025))))/(L - (-0.0099956095)*math.atan(L))) #1.5e-04
            return y
        def pr_y_V(L, d):
            y = ((-(-0.12630376*L*L/(L/math.cos(L) + 0.19825757) + (0.001852162 - 0.00014447162/L)/(L*L)) - 0.06392621)/(L*L)) #3.8e-04
            return y
        def pv_x_V(L, d):
            y = ((math.sin(math.atan(L)) + 0.43953407*math.sin(-0.22335356/L)/L)/2.0870857 - math.atan(math.sin(L - 0.028144019)) - (-0.4978432) - 0.09923751/L) #1.2e-04
            return y
        def pv_y_V(L, d):
            y = ((0.012981842 - 0.033012997/L)/(L + (-0.27790195)*(math.atan(3.7446797*L*L) + (-0.005431808)*math.atan(math.exp(math.cos(-1.356612/L)))))) #3.8e-04
            return y
    else:
        # --- MARS ---
        def pr_x_M(a, d):
            denom = 7.780825627926492 * a - 0.34159993346898454
            sin_arg = 2 * np.pi * a + 0.1231174475945867 + d * 0.5853586584957071 / (a**2 - 1.5690074915306857 * a + 1.9819223815154108)
            exp_arg = -3.424188046763629 * a
            exp_term = d * (-41.26470185582318 + 86.35986725318767 * a - 51.08371071757919 * a**2) * np.exp(exp_arg)
            numerator = 0.20074211987250948
            result = numerator / (denom + np.sin(sin_arg) + exp_term)
            return result
    
        def pr_y_M(a, d):
            cos_arg = 2 * np.pi * a + d * (
                -2.7056930690628103 * a**2 + 5.176300671261158 * a - 2.336378597152512
            ) * np.exp(3.2003639491958467 * a - 3.2205106566099353 * a**2)
            numerator = 0.0015463527047873246 - 0.0031678016585249486 * np.cos(cos_arg)
            denom = a**2 - 0.18328217249773662 * a + 0.11514749689845788
            result = numerator / denom
            return result
    
        def pv_x_M(a, d):
            cos_arg = 2 * np.pi * a + d * (
                -7.277086750486235 * a**2 + 13.74021831427378 * a - 6.262552219285274
            ) * np.exp(1.381195073394106 * a - 2.278956114228774 * a**2)
            numerator = 0.0018661818762522732 - 0.0031537940448213170 * np.cos(cos_arg)
            denom = a**2 - 0.19152227381595716 * a + 0.0993237695309777
            result = numerator / denom
            return result
    
        def pv_y_M(a, d):
            denom = 4.050922375438426 * a - 0.15902042687671913
            sin_arg = 2 * np.pi * a + 0.11429217276196703 + d * 0.012218372217935064 / (a**2 - 2.8347986183365705 * a + 2.06664561825753)
            exp_arg = -5.877538308252977 * a
            exp_term = d * (-540.5981059117084 * a**2 + 958.0032106019531 * a - 415.15730979365355) * np.exp(exp_arg)
            numerator = 0.10456877009148134
            result = numerator / (denom + np.sin(sin_arg) + exp_term)
            return result
    
        # --- VENUS ---
        def pr_x_V(a, d):
            denom = 7.737402704800786 * a - 0.32876870077839637
            sin_arg = 2 * np.pi * a + 0.10377485561964216 + d * 0.5010768929874032 / (a**2 - 1.9514741381243845 * a + 2.4016982504424953)
            exp_arg = -3.086139681061194 * a
            exp_term = d * (-39.740029616989275 + 73.28781124258543 * a - 40.57760476908843 * a**2) * np.exp(exp_arg)
            numerator = -0.2489269721475946
            result = numerator / (denom + np.sin(sin_arg) + exp_term)
            return result
    
        def pr_y_V(a, d):
            cos_arg = 2 * np.pi * a + d * (
                -5.520942046733101*a**2 + 9.93390376988929*a - 4.541315615665989
            ) * np.exp(1.3808955605425999*a - 2.0000002462936357*a**2)
            numerator = -0.006779491667620585 + 0.00432410667098884 * np.cos(cos_arg)
            denom = a**2 + 0.08726450366262754*a - 0.20030536675890054
            result = numerator / denom
            return result
    
        def pv_x_V(a, d):
            cos_arg = 2*np.pi*a + d*(
                -1.786880982902473*a**2 + 3.1517137043764167*a - 1.3891212954291232
            )*np.exp(2.9860241498160076*a-2.699241528126778*a**2)
            numerator = -0.006319471111610397 + 0.004278503661785389*np.cos(cos_arg)
            denom = a**2+0.13118547856852233*a-0.24254361523409607
            result = numerator / denom
            return result
    
        def pv_y_V(a, d):
            denom = 4.0002764433454105*a-0.15629991248717665
            sin_arg = 2*np.pi*a+0.08483203399561368+d*0.008896394730198013/(a**2-2.9131929806240318*a+2.1648040548553382)
            exp_arg = -6.814266658923464*a
            exp_term = d*(-1354.631280683438*a**2+2264.6420824634483*a-951.3449842396692)*np.exp(exp_arg)
            numerator = -0.12867786064634879
            result = numerator / (denom + np.sin(sin_arg) + exp_term)
            return result
    k = np.clip(k, 1e-6,np.inf)
    k1=0.72
    k2=1.52

    # prx
    p1 = pr_x_V(a,d)
    p2 = pr_x_M(a,d)
    g = (p2*k2**(1/4)/np.log(k2) - p1*k1*k2**(-3/4)/np.log(k1))/(1-(k1/k2)**(3/4))
    f = p1*k1/np.log(k1)-g*k1**(3/4)
    pr_x = np.log(k)*(f/k+g*k**(-1/4))

    # pry
    p1 = pr_y_V(a,d)
    p2 = pr_y_M(a,d)
    g = (p2*k2**(3/2)/np.log(k2) - p1*k1*np.sqrt(k2)/np.log(k1))/(1-np.sqrt(k2/k1))
    f = p1*k1/np.log(k1)-g/np.sqrt(k1)
    pr_y = np.log(k)*(f/k+g/(k**(3/2)))

    # pvx
    p1 = pv_x_V(a,d)
    p2 = pv_x_M(a,d)
    g = (p2*k2**(3/2)/np.log(k2) - p1*k1*np.sqrt(k2)/np.log(k1))/(1-np.sqrt(k2/k1))
    f = p1*k1/np.log(k1)-g/np.sqrt(k1)
    pv_x = np.log(k)*(f/k+g/(k**(3/2)))

    # pvy
    p1 = pv_y_V(a,d)
    p2 = pv_y_M(a,d)
    g = (p2*k2**(1/4)/np.log(k2) - p1*k1*k2**(-3/4)/np.log(k1))/(1-(k1/k2)**(3/4))
    f = p1*k1/np.log(k1)-g*k1**(3/4)
    pv_y = np.log(k)*(f/k+g*k**(-1/4))

    pr_0 = np.array([pr_x, pr_y, 0])
    pv_0 = np.array([pv_x, pv_y, 0])
    return pr_0, pv_0

class OptimalControlSolver:
    '''
    Сlass for solving the optimal control problem using the sighting method
    '''
    def __init__(self, mu: float, a_max: float, alpha: float, func_type):
        self.mu = mu
        self.a_max = a_max
        self.alpha = alpha
        self.func_type = func_type

    def set_boundary_conditions(self, r0, v0, rf, vf, T):
        self.r0 = np.array(r0, dtype=np.float64)
        self.v0 = np.array(v0, dtype=np.float64)
        self.rf = np.array(rf, dtype=np.float64)
        self.vf = np.array(vf, dtype=np.float64)
        self.T = T

    def optimal_control(self, pv):
        pv_norm = np.linalg.norm(pv)

        if self.func_type == 'quadratic':
            a = pv

        elif self.func_type == 'linear':
            a = self.a_max * pv / pv_norm if pv_norm > 1e-15 else np.zeros(3)

        elif self.func_type == 'mixed':
            if abs(self.alpha) < 1e-10:
                a = pv
            elif abs(self.alpha - 1.0) < 1e-10:
                a = self.a_max * pv / pv_norm if pv_norm > 1e-10 else np.zeros(3)
            else:
                if pv_norm <= self.alpha:
                    a = np.zeros(3)
                else:
                    a = ((pv_norm - self.alpha) / (1 - self.alpha)) * (pv / pv_norm)
                    a_norm = np.linalg.norm(a)
                    if a_norm > self.a_max:
                        a = (self.a_max / a_norm) * a

        return a

    def system_equations(self, t, y):
        r = y[0:3]
        v = y[3:6]
        pr = y[6:9]
        pv = y[9:12]

        r_norm = np.linalg.norm(r)

        drdt = v
        a = self.optimal_control(pv)
        dvdt = -self.mu * r / r_norm**3 + a
        dot_pr = (self.mu / r_norm**3) * pv - (3 * self.mu / r_norm**5) * np.dot(r, pv) * r
        dot_pv = -pr

        return np.concatenate([drdt, dvdt, dot_pr, dot_pv])

    def integrate_trajectory(self, p0, t_span=None):
        if t_span is None:
            t_span = [0, self.T]

        y0 = np.concatenate([self.r0, self.v0, p0[0:3], p0[3:6]])
        sol = solve_ivp(
            self.system_equations,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10
        )
        return sol

    def loss(self, x):
        pr0 = x[0:3]
        pv0 = x[3:6]

        y0 = np.concatenate([self.r0, self.v0, pr0, pv0])
        sol = solve_ivp(
            self.system_equations,
            [0, self.T],
            y0,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10
        )

        if sol.success:
            rf = sol.y[0:3, -1]
            vf = sol.y[3:6, -1]
            r_res = self.rf - rf
            v_res = self.vf - vf
            return np.linalg.norm(np.hstack((r_res, v_res)))
        else:
            return 1e6

    def solve(self, p0_guess=None, method='Nelder-Mead', options=None):
        if p0_guess is None:
            pr_0, pv_0 = get_initial_adjoint(0.5, 1.2)
            p0_guess = np.hstack((pr_0, pv_0))

        if options is None:
            options = {'disp': False, 'maxiter': 40}

        result = minimize(
            self.loss,
            p0_guess,
            method=method,
            options=options
        )

        self.p0 = result.x
        self.trajectory = self.integrate_trajectory(self.p0)

        return result

    def continuation_method(self, alpha_values, p0_initial=None, a_max_fixed=0.3):
        self.a_max = a_max_fixed

        results = []
        current_p0 = p0_initial

        for alpha in alpha_values:
            self.alpha = alpha
            result = self.solve(current_p0)

            current_p0 = self.p0.copy()

            results.append({
                'alpha': alpha,
                'trajectory': self.trajectory
            })

        return results
    def compute_thrust_profile(self, trajectory):
        t = trajectory.t
        y = trajectory.y
        thrust = []

        for i in range(len(t)):
            pv = y[9:12, i]
            a = self.optimal_control(pv)
            thrust.append(np.linalg.norm(a))

        return t, np.array(thrust)


    def plot_trajectories_2d(self, results, filename='trajectories.png'):
        plt.figure(figsize=(10, 8))

        if results:
            colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

            for res, color in zip(results, colors):
                traj = res['trajectory']
                if traj is not None and traj.success:
                    r = traj.y[0:3]
                    plt.plot(r[0], r[1], color=color, linewidth=2,
                             alpha=0.7, label=f'α={res["alpha"]:.2f}')

        plt.scatter([self.r0[0]], [self.r0[1]], c='g', s=200, marker='o', label='Начало')
        plt.scatter([self.rf[0]], [self.rf[1]], c='r', s=200, marker='s', label='Цель')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Метод продолжения: множество траекторий при разных α')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        plt.savefig(filename, dpi=150)
        print(f"График сохранён: {filename}")
        plt.close()
    def plot_thrust_profiles(self, results, filename='thrust_profiles.png'):
        plt.figure(figsize=(10, 6))

        for res in results:
            traj = res['trajectory']
            alpha = res['alpha']

            t, thrust = self.compute_thrust_profile(traj)
            plt.plot(t, thrust, label=f'α={alpha:.2f}')

        plt.xlabel('t')
        plt.ylabel('|a(t)|')
        plt.title('Профиль тяги при разных α')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"График сохранён: {filename}")
        plt.close()



# пример
if __name__ == "__main__":
    solver = OptimalControlSolver(
        mu=1.0,
        a_max=1.0,
        alpha=0.0,
        func_type='mixed'
    )

    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    rf = np.array([-1.2, 0.0, 0.0])
    vf = np.array([0.0, -1/np.sqrt(1.2), 0.0])


    T = 2*np.pi*T_a(0.5,1.2)

    solver.set_boundary_conditions(r0, v0, rf, vf, T)
    alpha_values = np.linspace(0, 0.9, 20)

    results = solver.continuation_method(alpha_values, a_max_fixed=0.5)

    solver.plot_trajectories_2d(results, 'continuation_trajectories.png')
    solver.plot_thrust_profiles(results, 'thrust_profiles.png')

