from os import name
import numpy as np
import matplotlib.pyplot as plt

cb = np.array(0.1 * 0.9505)

va = 50

kw = 10 ** (-14)

def h2so4(ph: np.ndarray, graph: bool = False):
    k1 = np.array(10 ** (-2.35))
    k2 = np.array(10 ** (-1.99))

    beta1 = 10 ** (11.46)
    beta2 = 10 ** (17.89)
    beta3 = 10 ** (19.72)

    f = 1.000
    ca = np.array(0.01) * f

    h = 10 ** (-ph)
    oh = kw / h

    print(f"\npH:\n{ph}")

    da, dha, dh2a, dh3a = calc_mol(h, beta1, beta2, beta3)
    print(f"\nモル分率\nPO4:\n{da * 100}\nHPO4:\n{dha*100}\nH2PO4:\n{dh2a*100}\nH3PO4:\n{dh3a*100}")

    n = n_bar(dha, dh2a, dh3a)

    print(f"\n平均プロトン数:\n{n}")

    vb = calc_vb2(ca, va, n, cb, h, oh)

    print(f"\n滴下量:\n{vb}")

    alpha = calc_alpha(h, beta1, beta2, beta3)

    print(f"\n副反応係数:\n{alpha}")

    print(f"\nlog副反応係数:\n{np.log(alpha)}")
    print(f"\nlog副反応係数:\n{np.log(alpha) / 4}")

    a = calc_a(n, h, oh, ca)


    if graph:

        plt.plot(ph, vb)
        plt.ylim(0, 20)
        plt.title("vb")
        plt.show()

        plt.plot(ph, da, label="PO4")
        plt.plot(ph, dha, label="HPO4")
        plt.plot(ph, dh2a, label="H2PO4")
        plt.plot(ph, dh3a, label="H3PO4")
        plt.title("mol %")
        plt.legend()
        plt.show()

        plt.plot(ph, np.log(alpha))
        plt.title("log alpha")
        plt.show()

        plt.plot(ph, n)
        plt.title("n")
        plt.show()





def calc_mol(h, b1, b2, b3):
    a = calc_alpha(h, b1, b2, b3)
    da = 1 / a

    dha = b1 * h / a

    dh2a = b2 * (h**2) / a

    dh3a = b3 * (h ** 3) / a

    return da, dha, dh2a, dh3a

def n_bar(dha, dh2a, dh3a):
    return dha + 2 * dh2a + 3 * dh3a

def calc_vb(va, ca, dha, dh2a, h, oh, cb):
    """
    should not use
    """
    vb = (va * (2 * ca - dha - 2 * dh2a - h + oh)) / (cb + dha + 2 * dh2a + h - oh)
    return vb

def calc_vb2(ca, va, n, cb, h, oh):
    vb = (ca * va * (3 - n) - va * (h - oh)) / (cb + h - oh)
    return vb

def calc_alpha(h, b1, b2, b3):
    return 1 + b1 * h + b2 * (h ** 2) + b3 * (h ** 3)

def calc_a(n, h, oh, ca):
    return 2- n - (h - oh) / ca



def plot(x, y):
    plt.plot(x, y)
    plt.show()
    

def test():
    ph = np.linspace(0, 14)
    graph = True
    h2so4(ph, graph=graph)
    # h2so4(4)

def get_value():
    ph = np.array([0, 1, 1.5, 2, 2.23, 2.5, 3, 4, 5, 5.5, 6, 6.5, 7, 8, 9, 10, 11, 11.5, 11.75,12, 13])
    print(len(ph))
    h2so4(ph, graph=True)


if __name__ == "__main__":
    # test()
    get_value()
    # h2so4(4)