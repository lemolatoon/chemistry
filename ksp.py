import numpy as np


def one_exp(k=None):
    water = np.array([1, 5, 10]) # ml
    agno3 = np.array([8, 13, 15]) # teki

    c_ag = 3.00 * (10 ** (-3)) # mol / l

    #不明
    c_cl = 1.0987 * (10 ** (-4)) # mol / l

    if k is None:
        k = 0.08
        k = 0.064 #ml / teki
    print(f"\nk:{k}")


    #単位調整
    ag_l = agno3 * k * (10 ** (-3)) # l
    water = water * (10 ** (-3)) # l

    ag = ag_l * c_ag / (water + ag_l)
    cl = water * c_cl / (water + ag_l)

    ksp = ag * cl

    print(f"\nksp:\n{ksp}")
    print(f"\nksp_mean:\n{np.mean(ksp)}")


def ksp():
    c_cl = 1.0987 * (10 ** (-4))


def compare():
    one_exp(k=0.08)
    one_exp(k=0.064)

if __name__ == "__main__":
    compare()