import numpy as np
import scipy.interpolate as interpolate
from scipy import integrate
from decimal import *
import matplotlib.pyplot as plt
import warnings
import csv

# warnings.filterwarnings("ignore")
class Test():
    def __init__(self):
        self.n = 400
        self.xr1 = np.log(3.e-6)
        self.xr2 = np.log(60.e0)
        self.hy=0.1
        self.hr = (self.xr2-self.xr1)/(self.n-1)
        self.ymax=30.0 # maximum rapidity scale, termination of evolution
        self.y=np.arange(0.0,self.ymax+1,self.hy) # array of rapidity points to be evaluated over

        self.r_arr = np.zeros(400)
        self.xlr = np.zeros(400) # log space
        self.n_current = np.zeros(400) # data points of N(r,Y) at current evolution
        self.r = 0.0

        # precision parameters
        self.xprec1=5e-3
        self.xprec2=5e-4
        self.xprec3=1.5e-2

        self.nmax1=2000
        self.nmax2=20000
        self.rlim1=5e-8
        self.rlim2=0.9999

        self.xlam=0.241 # lambda QCD
        self.af=1.0 # frozen alpha
        self.ge=0.5772156659 # Euler gamma
        self.xnf=3.0 # number of active flavors
        self.beta=(33-2.0*self.xnf)/(12.0*np.pi)
        self.pre=0.5/self.beta
        self.auxal=2.0/self.xlam
        self.iglob=1

        self.xn=60.e0
        self.qsq2=0.168e0*self.xn
        self.gamma=1.0

        self.method="RK4"

        self.xk = np.zeros((4,self.n))

        for i in range(400):
            self.xlr[i] = self.xr1+i*self.hr
            self.r_arr[i] = np.exp(self.xlr[i])
            xlog = np.log(1.0/(0.241*self.r_arr[i])+np.exp(1.0))
            xexp = np.power(self.qsq2*np.power(self.r_arr[i],2),self.gamma)*xlog/4.0
            self.n_current[i] = 1.0-np.exp(-xexp)
        print("r array")
        print(self.r_arr)

  #      self.func = interpolate.interp1d(self.xlr, self.n_current, kind=5)
        self.t_ = 0.0
        self.func = None

    def alphakw(self, r):
        cte = np.exp(1.0/(self.beta*self.af))
        alpha = 2.0*self.pre/np.log(np.power(self.auxal/r,2) + cte)

        return alpha

    def interpolate_n(self, x):
        n_ = 0.0
        if x < self.xr1:
            ex = 1.99
            n_ = np.power(self.n_current[0]*np.exp(x)/self.r_arr[0], ex)
        elif x > self.xr2:
            n_ = 1.0
        else:
            n_ = self.func(x)

        if n_ < 0.0: return 0.0
        if n_ > 1.0: return 1.0

        return n_

# simpson integrator, w/ corresponding kernel and integrand
    def ker_gq(self, rvar, r1, r2):
        if r1 < 1.e-20 or r2<1.e-20:
            return 0.0
        else:
            pre = 3.0/(2.0*np.power(np.pi, 2))
            rr = np.power(rvar, 2)
            r12 = np.power(r1, 2)
            r22 = np.power(r2, 2)

            ar = self.alphakw(rvar)
            a1 = self.alphakw(r1)
            a2 = self.alphakw(r2)

            t1 = rr/(r12*r22)
            t2 = (a1/a2 - 1.0)/r12
            t3 = (a2/a1 - 1.0)/r22

            return pre*ar*(t1+t2+t3)
# integrate function f (gaussian quadrature) for inner bounds [a,b], outer bounds [c,d], to precision prec

    def simpson(self, f, a, b, tol):
# IMPLEMENT NEW INTEGRATOR
        return 0.0

    """ def intg_gq(self, f, a, b, tol):
        getcontext().prec = 20
        # weights and abscissas for Gaussian quadrature at n=8:
        w = [Decimal(0.101228536290376259), Decimal(0.222381034453374471),Decimal(0.313706645877887287),Decimal(0.362683783378361983),
             Decimal(0.0271524594117540949),Decimal(0.0622535239386478929),Decimal(0.0951585116824927848),Decimal(0.124628971255533872),
             Decimal(0.149595988816576732),Decimal(0.169156519395002538),Decimal(0.182603415044923589),Decimal(0.189450610455068496)]

        x = [Decimal(0.960289856497536232),Decimal(0.796666477413626740),Decimal(0.525532409916328986),Decimal(0.183434642495649805),
             Decimal(0.989400934991649933),Decimal(0.944575023073232576),Decimal(0.865631202387831744),Decimal(0.755404408355003034),
             Decimal(0.617876244402643748),Decimal(0.458016777657227386),Decimal(0.281603550779258913),Decimal(0.0950125098376374402)]

#        print("a="+str(a)+" b="+str(b)+" tol="+str(tol))
        s = Decimal(0.0)
        iter=0.0
        cont1 = True
        if b == a: return s
        const = 0.005 / (b - a)
        bb = a

        while cont1:
            aa = bb
            bb = b

            iter += 1
            cont = True
            while cont:
                c1 = Decimal(0.5 * (float(bb) + aa))
                c2 = Decimal(0.5 * (float(bb) - aa))
                s8 = Decimal(0.0)

                for i in range(4):
                    u = Decimal(c2) * x[i]
                    s8 += w[i] * Decimal(f(float(c1 + u)) + f(float(c1 - u)))

                s8 = c2 * s8
                s16 = Decimal(0.0)

                for i in range(4, 12):
                    u = c2 * x[i]
                    s16 += w[i] * Decimal(f(float(c1 + u)) + f(float(c1 - u)))

                s16 = c2 * s16

                print("lhs="+str(np.abs(s16-s8))+" rhs="+str(Decimal(tol)*np.abs(s16)))

                if np.abs(s16 - s8) <= Decimal(tol) * np.abs(s16):
                    cont = False
                bb = c1
                if (Decimal(1.0) + Decimal(np.abs(Decimal(const) * c2))) != 1.0:
                    cont = True

            s += s16
            if iter > self.nmax2: iter = 0.0
            if bb != b:
                cont1 = True
            else:
                cont1 = False
        print("s="+str(s))
        return s
"""
# function inner_gq gets integrated over variable xlz
    def inner_integrand_gq(self,xlz):
        z = np.exp(xlz)
        zz = z*z

        q12 = 0.25*np.power(self.r,2)+zz-self.r*z*np.cos(self.t_)
        q22 = 0.25*np.power(self.r,2)+zz+self.r*z*np.cos(self.t_)

        if q12 < 0.0: q12=0.0
        if q22 < 0.0: q22=0.0

        q1 = np.sqrt(q12)
        q2 = np.sqrt(q22)

        fr0 = self.n_current[np.where(self.r_arr == self.r)]
        fr1 = self.interpolate_n(np.log(q1))
        fr2 = self.interpolate_n(np.log(q2))

        ker = self.ker_gq(self.r, q1, q2)
        # print("zz="+str(zz)+" ker="+str(ker)+" fr0="+str(fr0)+" fr1="+str(fr1)+" fr2="+str(fr2))
        return zz*ker*(fr1+fr2-fr0-fr1*fr2)[0]

# define self.r
    def outer_integrand_gq(self, tvar):
        self.t_ = tvar
        vfr = self.n_current[np.where(self.r_arr == self.r)]

        if vfr < self.rlim1:
            xprec = self.xprec1/10.0
        elif vfr > self.rlim2:
            xprec = self.xprec3/10.0
        else:
            xprec = self.xprec2/20.0
       # print("inner integral called")
       # print("xr1="+str(self.xr1)+" xr2="+str(self.xr2))
       # integral = self.intg_gq(self.inner_integrand_gq, self.xr1, self.xr2, xprec)

        # integral = integrate.quadrature(self.inner_integrand_gq, self.xr1, self.xr2, tol=xprec, vec_func=False)[0]
        integral = integrate.quad(self.inner_integrand_gq, self.xr1, self.xr2,epsabs=xprec)[0]
        # integral = integrate.romberg(self.inner_integrand_gq, self.xr1, self.xr2, tol=xprec)
        # print("inner = " + str(integral))
        return integral

    """  def xk(self):
        integral = np.zeros(400)

        for i in range(len(self.r_arr)):
            self.r = self.r_arr[i]
            vfr = self.n_current[i]

            if vfr < self.rlim1:
                xprec = self.xprec1
            elif vfr >self.rlim2:
                xprec = self.xprec3
            else:
                xprec = self.xprec2

            # integral[i] = 4.0*self.intg_gq(self.outer_integrand_gq, 0.0, 0.5*np.pi, xprec)
            integral[i] = 4.0*integrate.quadrature(self.outer_integrand_gq,0.0,0.5*np.pi, tol=xprec, vec_func=False)[0]

            # integral[i] = 4.0 * integrate.romberg(self.outer_integrand_gq,0.0,0.5*np.pi, tol=xprec)
        # for i in range(self.n):
        #     print(integral[i])
        return integral
"""
    def evolve(self):
        # 4th order runge-kutta, and then call interpolations after correction for self.n
        with open("result.csv","w") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")

            with open("result2.csv","w") as csvfile2:
                writer2 = csv.writer(csvfile2,delimiter="\t")
                for i in range(len(self.y)):
                    y0 = self.y[i]
                    print("Y=", y0)

                    self.func = interpolate.interp1d(self.xlr, self.n_current, kind=5)

                    if self.method == "Euler": kkuta = 1
                    elif self.method == "RK2": kkuta = 2
                    elif self.method == "RK4": kkuta = 4

        #            n_correction = self.xk()
        #            n_corrected = self.n_current

                    aux_n = self.n_current # creates auxiliary array, copy of n_current
                    dy = 0.1

                    for k in np.arange(1, kkuta+1):
                        print("KUTTA " + str(k))
                        if kkuta == 2:
                            if k == 1: dy = 0.5*self.hy
                            if k == 2: dy = self.hy
                        elif kkuta == 4:
                            if k == 1: dy = 0.5*self.hy
                            if k == 2: dy = 0.5*self.hy
                            if k == 3: dy = self.hy
                            if k == 4: dy = self.hy
                        """
                        for j in range(self.n): # 737
                            n_corrected[j] += dy*n_correction[j]
                            if n_corrected[j] > 1.0:
                                n_corrected[j] = 1.0
                        # 717
        
                        n_next = np.zeros(self.n)
                        for i in range(self.n):
                            n_next[i] = self.n_current[i]+self.hy*n_correction[i]
                            if n_next[i] > 1.0:
                                n_next[i] = 1.0
        
                        self.n_current = n_next
        """
                        prev = 0.0 # ???
                        for i in range(self.n):
                            self.r = self.r_arr[i]
                            vfr = aux_n[i]

                            if (vfr > self.rlim2) and (np.abs(1.0-prev) < 1.e-13):
                                self.xk[k-1,i] = 0.0
                            else:
                                if vfr < self.rlim1:
                                    xprec=self.xprec1
                                elif vfr > self.rlim2:
                                    xprec=self.xprec3
                                else:
                                    xprec=self.xprec2

                                # self.xk[k-1,i] = 4.0*integrate.quadrature(self.outer_integrand_gq, 0.0, 0.5*np.pi, tol = xprec, vec_func=False)[0]
                                self.xk[k - 1, i] = 4.0 * integrate.quad(self.outer_integrand_gq, 0.0, 0.5 * np.pi,epsabs=xprec)[0]
                                if k == 4: print("{:e}".format(self.xk[k-1, i]))
                            prev = (vfr+dy*self.xk[k-1, i])/vfr

                        for i in range(self.n):
                            aux_n[i] = self.n_current[i] + dy*self.xk[k-1, i]
                            if aux_n[i] > 1.0:
                                aux_n[i]= 1.0
                            writer.writerow([y0,self.r_arr[i],aux_n[i]])

                    for i in range(self.n):
                        self.n_current[i] = self.n_current[i] + self.hy*self.xk[2,i]
                        if self.n_current[i] > 1.0:
                            self.n_current[i] = 1.0
                        writer2.writerow([y0,self.r_arr[i],self.n_current[i]])


if __name__ == "__main__":
    t = Test()
    t.evolve()




