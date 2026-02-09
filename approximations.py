import numpy as np
import math


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