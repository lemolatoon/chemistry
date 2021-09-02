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
        v_ag = tf.constant(10.0, dtype=tf.float64) # メスシリンダー

        for val in (v_w, x, k, ksp, k1, v_ag):
            for tape in tapes:
                tape.watch(val)

        c_ag = tf.constant(0.1, dtype=tf.float64) * k1 /(v_ag + k1)

        print(f"{c_ag}")


        c_cl = ksp * (v_w + x * k) ** 2 / (c_ag * k * v_w * x)


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

    delta_x = 0.5 * x
    delta_vw= 0.01
    delta_k = 0.01
    delta_k1 = 0.01
    delta_vag = 0.01

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

    print(f"\nx(滴数)の誤差\n{delta_by_x}")
    print(f"\nvw(滴下先の水)の誤差\n{delta_by_vw}")
    print(f"\nk(3ml駒込)の誤差\n{delta_by_k}")
    print(f"\nk1(滴瓶)の誤差\n{delta_by_k1}")
    print(f"\nvag(agno3の希釈時の水)の誤差\n{delta_by_vag}")

    delta = delta_by_x + delta_by_vw + delta_by_k + delta_by_k1 + delta_by_vag

    print(f"\n誤差合計:\n{delta}")

    print(f"\n滴数の誤差平均\n{tf.reduce_mean(delta)}")
    print(f"{c_cl + tf.reduce_mean(delta)}")
    print(f"{c_cl + delta}")



def calc_ksp(k=None):
    print("\n==calc==")
    V0 = 50 + 0.055 # ml

    c_cl = 1.0987 * (10 ** (-4))
    c_ag = 3.00 * (10 ** (-3))

    x = np.array([12, 19, 29]) # teki

    if k is None:
        k = 0.08 # ml / teki
        #やはり3ml駒込
    print(f"k:{k}\n")
    T = [40, 60, 80]

    ksp = (V0 * c_cl * c_ag * x * k) / ((V0 + x * k) ** 2)

    print(f"温度:\n{T}")
    print(f"ksp:\n{ksp}")


def compare():
    f = calc_ksp
    f(k=0.08)
    f(k=0.064)

if __name__ == "__main__":
    #compare()
    #calc_ksp()
    #one_exp()
    c_cl_gradient()