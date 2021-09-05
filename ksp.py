from typing import Tuple
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf



v_w = np.array([1, 5, 10]) # ml
x = np.array([8, 13, 15]) # teki

c_ag = np.array(3.00 * (10 ** (-3))) # mol / l

k = 0.08

ksp = np.array(1.7 * (10 ** (-10)))


def one_exp(k=None):
    v_w = np.array([1, 5, 10]) # ml
    x = np.array([8, 13, 15]) # teki

    c_ag = 3.00 * (10 ** (-3)) # mol / l

    #不明
    c_cl = 1.0987 * (10 ** (-4)) # mol / l

    if k is None:
        k = 0.064 #ml / teki
        k = 0.08
    print(f"\nk:{k}")

    ksp = 1.7 * (10 ** (-10))
    #単位調整

    c_cl = ksp * (v_w + x * k) ** 2 / (c_ag * k * v_w * x)


    print(f"\nC_Cl:\n{c_cl}")
    print(f"\nmean_C_Cl:\n{np.mean(c_cl)}")

def c_cl_gradient(v_w=v_w, x=x, c_ag=c_ag, k=k, ksp=ksp):
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3, tf.GradientTape() as tape4, tf.GradientTape() as tape5:
        tapes: Tuple[tf.GradientTape, tf.GradientTape] = (tape1, tape2, tape3, tape4, tape5) 
        v_w = tf.convert_to_tensor(v_w, dtype=tf.float64) # メスシリンダー
        x = tf.convert_to_tensor(x, dtype=tf.float64) # 滴数
        k = np.full((3,), k)
        k = tf.convert_to_tensor(k, dtype=tf.float64) # 3ml駒込一滴あたりのml
        ksp = tf.convert_to_tensor(ksp) # 文献値

        #c_ag = tf.convert_to_tensor(c_ag) # 
        k1 = np.full((3,), 0.05)
        k1 = tf.convert_to_tensor(k1, dtype=tf.float64) # 滴瓶一滴あたりのml
        v_ag = tf.convert_to_tensor(np.full((3,), 10.0), dtype=tf.float64) # メスシリンダー

        for val in (v_w, x, k, ksp, k1, v_ag):
            for tape in tapes:
                tape.watch(val)

        c0 = tf.constant(0.1, dtype=tf.float64)
        c_ag = tf.constant(0.1, dtype=tf.float64) * k1 /(v_ag + k1)

        print(f"{c_ag}")

        print(f"vag:{v_ag}, k1:{k1}, k2:{k}, ksp:{ksp}, x:{x}, v_w{v_w}")

        c_cl = (v_ag + k1) * ((v_w + x * k) ** 2) * ksp / (0.1 * k1 * k * v_w * x)
        #print(c_cl)

        #c_cl = ksp * (v_ag + k1) * ((v_w + x * k) ** 2) / (c_ag * k * v_w * x)


    print(f"\nC_Cl:\n{c_cl}")
    print(f"mean_C_Cl:\n{np.mean(c_cl)}")

    dc_dx = tape1.gradient(c_cl, x)
    dc_dvw = tape2.gradient(c_cl, v_w)
    dc_dk = tape3.gradient(c_cl, k)
    dc_dk1 = tape4.gradient(c_cl, k1)
    dc_dvag = tape5.gradient(c_cl, v_ag)

    print(f"\ndc_dx:\n{dc_dx}")
    print(f"dc_dvw:\n{dc_dvw}")
    print(f"dc_dk:\n{dc_dk}")
    print(f"dc_dk1:\n{dc_dk1}")
    print(f"dc_dvag:\n{dc_dvag}")

    delta_x = 1
    delta_vw= 0.005
    delta_k = 0.01
    delta_k1 = 0.01
    delta_vag = 0.005

    delta_by_x = dc_dx * delta_x
    delta_by_vw = dc_dvw * delta_vw
    delta_by_k = dc_dk * delta_k
    delta_by_k1 = dc_dk1 * delta_k1
    delta_by_vag = dc_dvag * delta_vag

    delta_by_x = tf.abs(delta_by_x)
    delta_by_vw = tf.abs(delta_by_vw)
    delta_by_k = tf.abs(delta_by_k)
    delta_by_k1 = tf.abs(delta_by_k1)
    delta_by_vag = tf.abs(delta_by_vag)

    print(f"\nx(滴数)の誤差\n{delta_by_x}\n{tf.reduce_mean(delta_by_x)}")
    print(f"\nvw(滴下先の水)の誤差\n{delta_by_vw}\n{tf.reduce_mean(delta_by_vw)}")
    print(f"\nk(3ml駒込)の誤差\n{delta_by_k}\n{tf.reduce_mean(delta_by_k)}")
    print(f"\nk1(滴瓶)の誤差\n{delta_by_k1}\n{tf.reduce_mean(delta_by_k1)}")
    print(f"\nvag(agno3の希釈時の水)の誤差\n{delta_by_vag}\n{tf.reduce_mean(delta_by_vag)}")

    delta = delta_by_x + delta_by_vw + delta_by_k + delta_by_k1 + delta_by_vag

    print(f"\n誤差合計:\n{delta}")

    print(f"\n総和の誤差平均\n{tf.reduce_mean(delta)}")
    print(f"{c_cl + tf.reduce_mean(delta)}")
    print(f"delta     :{delta}")
    print(f"c_cl      :{c_cl}")
    print(f"c_cl+delta:{c_cl + delta}")
    print(f"c_cl-delta:{c_cl - delta}")



def calc_ksp(k=None, tapes: Tuple[tf.GradientTape]=None):
    print("\n==calc==")
    V0 = 50 + 0.055 # ml

    c_cl = 1.0987 * (10 ** (-4))
    c_ag = 3.00 * (10 ** (-3))

    x = np.array([12, 19, 29]) # teki

    if k is None:
        k = 0.08 # ml / teki
        #やはり3ml駒込
    print(f"k:{k}\n")
    T = np.array([38.6, 64.5, 79.3])

    if tapes is not None:
        for val in (V0, c_cl, c_ag, x, k):
            for tape in tapes:
                if val is not tf.Tensor:
                    val = tf.convert_to_tensor(val)
                tape.watch(val)

    ksp = (V0 * c_cl * c_ag * x * k) / ((V0 + x * k) ** 2)

    print(f"温度:\n{T}")
    print(f"ksp:\n{ksp}")

    return ksp


def calc_ksp_grad():
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3, tf.GradientTape() as tape4, tf.GradientTape() as tape5:
        tapes = (tape1, tape2, tape3, tape4, tape5)
        print("\n==calc==")
        V0 = np.array(50 + 0.055)
        V0 = np.full((3,), V0)
        V0 = tf.convert_to_tensor(V0, dtype=tf.float64)

        c_cl = np.array(1.0987 * (10 ** (-4)))
        c_cl = np.full((3,), c_cl)
        c_cl = tf.convert_to_tensor(c_cl, dtype=tf.float64)

        # c_ag = 3.00 * (10 ** (-3))
        k1 = np.array(0.05)
        k1 = np.full((3,), k1)
        k1 = tf.convert_to_tensor(k1, dtype=tf.float64)

        k2 = np.array(0.08)
        k2 = np.full((3,), k2)
        k2 = tf.convert_to_tensor(k2, dtype=tf.float64)

        Vag = np.array(5)
        Vag = np.full((3,), Vag)
        Vag = tf.convert_to_tensor(Vag, dtype=tf.float64)

        x = np.array([12, 19, 29]) # teki
        x = tf.convert_to_tensor(x, dtype=tf.float64)

        for val in (x, V0, c_cl, k1, k2, Vag):
            print(val)
            for tape in tapes:
                tape.watch(val)

        ksp = (V0 * c_cl * (0.3 * k1) * x * k2) / (Vag * (V0 + x * k2) ** 2)

    T = [40, 60, 80]
    print(f"温度:\n{T}")
    print(f"ksp:\n{ksp}")

    dksp_dx = tape1.gradient(ksp, x)
    dksp_dV0 = tape2.gradient(ksp, V0)
    dksp_dk1 = tape3.gradient(ksp, k1)
    dksp_dk2 = tape4.gradient(ksp, k2)
    dksp_dVag = tape5.gradient(ksp, Vag)

    print(f"\ndksp_dx:\n{dksp_dx}")
    print(f"dksp_dV0:\n{dksp_dV0}")
    print(f"dksp_dk1:\n{dksp_dk1}")
    print(f"dksp_dk2:\n{dksp_dk2}")
    print(f"dksp_dVag:\n{dksp_dVag}")

    val = 0.3 * c_cl * k1 * k2 * V0 * (V0 - k2 * x) / (Vag * (V0 + k2 * x) ** 3)
    print(f"val:{val}")

    delta_x = 1
    delta_V0 = 0.05
    delta_k1 = 0.01
    delta_k2 = 0.01
    delta_Vag = 0.005

    delta_by_x = dksp_dx * delta_x
    delta_by_V0 = dksp_dV0 * delta_V0
    delta_by_k1 = dksp_dk1 * delta_k1
    delta_by_k2 = dksp_dk2 * delta_k2
    delta_by_Vag = dksp_dVag * delta_Vag

    delta_by_x = tf.abs(delta_by_x)
    delta_by_V0 = tf.abs(delta_by_V0)
    delta_by_k1 = tf.abs(delta_by_k1)
    delta_by_k2 = tf.abs(delta_by_k2)
    delta_by_Vag = tf.abs(delta_by_Vag)

    print(f"\nxの誤差\n{delta_by_x}")
    print(f"{tf.reduce_mean(delta_by_x)}")
    print(f"V0の誤差\n{delta_by_V0}")
    print(f"{tf.reduce_mean(delta_by_V0)}")
    print(f"k1の誤差\n{delta_by_k1}")
    print(f"{tf.reduce_mean(delta_by_k1)}")
    print(f"k2の誤差\n{delta_by_k2}")
    print(f"{tf.reduce_mean(delta_by_k2)}")
    print(f"Vagの誤差\n{delta_by_Vag}")
    print(f"{tf.reduce_mean(delta_by_Vag)}")

    delta = delta_by_x + delta_by_V0 + delta_by_k1 + delta_by_k2 + delta_by_Vag

    print(f"\n誤差総和\n{delta}")
    print(f"\n誤差総和平均\n{tf.reduce_mean(delta)}")

    print(f"\ndelta    :{delta}")
    print(f"ksp      :{ksp}")
    print(f"ksp+delta:{ksp + delta}")
    print(f"ksp-delta:{ksp - delta}")



def compare():
    f = calc_ksp
    f(k=0.08)
    f(k=0.064)

def gradient():
    f = calc_ksp
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        tapes = (tape1, tape2)
        tapes = f(tapes=tapes)
        tape1, tape2 = tapes
    
    dksp_dV0 = tape1.gradient(ksp)
    print(dksp_dV0)

def fitting():
    T = np.array([38.6, 64.5, 79.3])
    ksp = calc_ksp()
    y = np.log(ksp)
    x = 1 / (273 + T)

    print(x)
    print(y)
    print()
    print((x - 2.7 * (10 ** (-3))) * 3 * (10 ** 3) * 10)
    print()
    print()
    a, b = np.polyfit(x, y, 1)

    print(f"a:{a}")
    print(f"b:{b}")

    import matplotlib.pyplot as plt

    #x =  np.arange(start=int(np.min(x)) - 1, stop=)
    

if __name__ == "__main__":
    # compare()
    #calc_ksp()
    fitting()
    # gradient()
    #calc_ksp_grad()
    #one_exp()
    #c_cl_gradient()