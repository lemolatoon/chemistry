import numpy as np
import matplotlib.pyplot as plt

cb = np.array(0.1 * 0.9505)

va = 50

kw = 10 ** (-14)

def h2so4(ph: np.ndarray, graph: bool = False):
    k2 = np.array(10 ** (-1.99))

    f = 1.000
    ca = np.array(0.004) * f

    h = 10 ** (-ph)
    oh = kw / h

    print(f"\npH:\n{ph}")

    da, dha, dh2a = calc_mol(h, k2)
    print(f"\nモル分率\nSO4:\n{da}\nHSO4:\n{dha}\nH2SO4:\n{dh2a}")

    n = n_bar(dha, dh2a)

    print(f"\n平均プロトン数:\n{n}")

    vb = calc_vb(va, ca, dha, dh2a, h, oh, cb)

    print(f"\n滴下量:\n{vb}")

    alpha = calc_alpha(h, k2)

    print(f"\n副反応係数:\n{alpha}")

    print(f"\nlog副反応係数:\n{np.log(alpha)}")

    a = calc_a(n, h, oh, ca)

    if graph:

        plt.plot(ph, vb)
        plt.ylim(0, 20)
        plt.show()

        plt.plot(ph, da)
        plt.plot(ph, dha)
        plt.plot(ph, dh2a)
        plt.show()


def h2so4_2(ph: np.ndarray, graph: bool = False):
    k2 = np.array(10 ** (-1.99))

    f = 1.000
    ca = np.array(0.004) * f

    h = 10 ** (-ph)
    oh = kw / h

    print(f"\npH:\n{ph}")

    da, dha, dh2a = calc_mol(h, k2)
    print(f"\nモル分率\nSO4:\n{da}\nHSO4:\n{dha}\nH2SO4:\n{dh2a}")


    n = n_bar(dha, dh2a)

    print(f"\n平均プロトン数:\n{n}")

    vb1 = calc_vb(va, ca, dha, dh2a, h, oh, cb)
    vb1 = calc_vb2(ca, va, n, cb, h, oh)



    # print(f"\n滴下量:\n{vb}")

    alpha = calc_alpha(h, k2)

    print(f"\n副反応係数:\n{alpha}")

    print(f"\nlog副反応係数:\n{np.log(alpha)}")

    a = calc_a(n, h, oh, ca)
    # a = 2 - n - (h - oh) / ca
    vb = a * (ca * va / cb)

    print("vb diff==================")
    print(vb1 - vb)

    if graph:

        plt.plot(ph, vb)
        plt.plot(ph, vb1)
        plt.ylim(0, 20)
        plt.show()
        
        plt.plot(ph, a)
        plt.ylim(0, 3)
        plt.show()
    
    



def calc_mol(h, k2):
    da = k2 / (k2 + h) 

    dha = h / (k2 + h)

    dh2a = np.full_like(da, 0)

    return da, dha, dh2a

def n_bar(dha, dh2a):
    return dha + 2 * dh2a

def calc_vb(va, ca, dha, dh2a, h, oh, cb):
    vb = (va * (2 * ca - dha - 2 * dh2a - h + oh)) / (cb + dha + 2 * dh2a + h - oh)
    return vb

def calc_vb2(ca, va, n, cb, h, oh):
    vb = (ca * va * (2 - n) - va * (h - oh)) / (cb + h - oh)
    return vb

def calc_alpha(h, k2):
    return 1 + h / k2

def calc_a(n, h, oh, ca):
    return 2- n - (h - oh) / ca



def plot(x, y):
    plt.plot(x, y)
    plt.show()
    

def test():
    ph = np.linspace(0, 14)
    h2so4(ph)
    # h2so4(4)


if __name__ == "__main__":
    test()