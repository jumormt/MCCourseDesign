# -*- coding: utf-8 -*-
'''
Created on 2017-08-05

@author: naiive
'''


'''
This is the channel encoding comparing that write by naiive.
'''

import numpy as np
import matplotlib.pylab as plb


plb.mpl.rcParams['font.sans-serif'] = ['SimHei']
plb.mpl.rcParams['axes.unicode_minus'] = False

mul = 100
pos = np.array([+1 ] *mul)
neg = np.array([-1 ] *mul)

syb = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
SIGS_syb = []

# SET 1
syb_a1 = np.array([pos, pos, pos, pos, pos, pos]  )# +1 +1 +1 +1 +1 +1 00
syb_b1 = np.array([neg, neg, neg, pos, pos, pos]  )# -1 -1 -1 +1 +1 +1 10
syb_c1 = np.array([pos, pos, pos, neg, neg, neg]  )# +1 +1 +1 -1 -1 -1 01
syb_d1 = np.array([neg, neg, neg, neg, neg, neg]  )# -1 -1 -1 -1 -1 -1 11
syb1 = [syb_a1, syb_b1, syb_c1, syb_d1]

sybj_a1 = np.array([+mul, +mul, +mul, +mul, +mul, +mul])
sybj_b1 = np.array([-mul, -mul, -mul, +mul, +mul, +mul])
sybj_c1 = np.array([+mul, +mul, +mul, -mul, -mul, -mul])
sybj_d1 = np.array([-mul, -mul, -mul, -mul, -mul, -mul])
sybj1 = [sybj_a1, sybj_b1, sybj_c1, sybj_d1]


SIGS1_syb = []
SIGR1_syb = []
SIGJ1_syb = []
BER1 = 0

# SET 2
syb_a2 = np.array([pos, pos, neg, neg, neg, neg]  )# +1 +1 -1 -1 -1 -1 00
syb_b2 = np.array([neg, neg, pos, pos, neg, neg]  )# -1 -1 +1 +1 -1 -1 10
syb_c2 = np.array([neg, neg, neg, neg, pos, pos]  )# -1 -1 -1 -1 +1 +1 01
syb_d2 = np.array([pos, pos, pos, pos, pos, pos]  )# +1 +1 +1 +1 +1 +1 11
syb2 = [syb_a2, syb_b2, syb_c2, syb_d2]

sybj_a2 = np.array([+mul, +mul, -mul, -mul, -mul, -mul])
sybj_b2 = np.array([-mul, -mul, +mul, +mul, -mul, -mul])
sybj_c2 = np.array([-mul, -mul, -mul, -mul, +mul, +mul])
sybj_d2 = np.array([+mul, +mul, +mul, +mul, +mul, +mul])
sybj2 = [sybj_a2, sybj_b2, sybj_c2, sybj_d2]

SIGS2_syb = []
SIGR2_syb = []
SIGJ2_syb = []
BER2 = 0

SIGtmp = []


# 产生num行6列，均值为0，方差为var的AWGN噪声
def generate_AWGN(var, num):
    AWGN_syb = []
    for i in range(num)  :  # num行6列,每列都是1行mul列的数组
        AWGN_syb.append(np.random.normal(0, var, size = (6, mul)))
    return AWGN_syb


# 产生num行6列的信号
def generate_signal(num):
    SIGS1_syb = []

    SIGS2_syb = []

    SIGS_syb = []

    SIGtmp = np.zeros((num, 6))

    p = np.random.randint(4, size=num  )  # 0-3的num个随机数
    for i in p:
        SIGS1_syb.append(syb1[i])
        SIGS2_syb.append(syb2[i])
        SIGS_syb.append(syb[i])

    SIGS1_syb = np.array(SIGS1_syb)
    SIGS2_syb = np.array(SIGS2_syb)

    return SIGS1_syb, SIGS2_syb, SIGtmp, SIGS_syb


# 添加高斯白噪声，模拟信号传播过程
def add_AWGN(SIGS_, AWGN_):
    SIGR_ = SIGS_ + AWGN_
    return SIGR_


# 绘制信号
def plot_SIG(SIGS, SIGR, title, Tnum):
    sigsx = []
    sigsy = []
    for i in SIGS:
        for j in i:
            for k in j:
                sigsy.append(k)
    for x in range(len(sigsy)):
        sigsx.append(x)
    sigrx = []
    sigry = []
    for i in SIGR:
        for j in i:
            for k in j:
                sigry.append(k)
    for x in range(len(sigry)):
        sigrx.append(x)
    plb.subplot(211)
    plb.plot(sigsx, sigsy, color='b', label=u'发送端')
    plb.ylabel('发送端')
    plb.xlim(0.0, 6* Tnum * mul)  # set axis limits

    plb.subplot(212)
    plb.plot(sigrx, sigry, color='r', label=u'接收端')  # use pylab to plot x and y
    plb.ylabel('接收端')
    plb.xlim(0.0, 6 * Tnum * mul)  # set axis limits
    plb.show()


# 接收端进行判决(软判决)
def get_SIGJ(SIGS_syb, SIGR_syb, sybj, SIGtmp):
    SIGJ_syb = SIGS_syb.copy()
    SIGJ_syb = np.array(SIGJ_syb)
    for i in range(len(SIGR_syb)):
        for j in range(6):
            SIGtmp[i][j] = np.sum(SIGR_syb[i][j])
    for obj in range(len(SIGJ_syb)):
        olen1 = np.sqrt(np.sum(np.square(SIGtmp[obj] - sybj[0])))
        olen2 = np.sqrt(np.sum(np.square(SIGtmp[obj] - sybj[1])))
        olen3 = np.sqrt(np.sum(np.square(SIGtmp[obj] - sybj[2])))
        olen4 = np.sqrt(np.sum(np.square(SIGtmp[obj] - sybj[3])))
        a = {olen1: np.array([0, 0]), olen2: np.array([1, 0]), olen3: np.array([0, 1]), olen4: np.array([1, 1])}
        olent1 = min([olen1, olen2, olen3, olen4])
        SIGJ_syb[obj] = a[olent1]
    SIGJ_syb = np.array(SIGJ_syb)
    return SIGJ_syb


# 计算误比特率
def get_BER(SIGS_syb, SIGJ_syb):
    count = 0
    for i in range(len(SIGJ_syb)):
        for k in ((SIGJ_syb[i] - SIGS_syb[i])):
            if (k != 0):
                count = count + 1
    BER = count / (2 * len(SIGS_syb))
    return BER


# 绘制误比特率图
def plot_BER():
    BERarr1 = []
    BERarr2 = []
    var1 = np.arange(3, 30, 0.5)
    var2 = np.arange(30, 200, 10)
    var3 = np.arange(200, 2200, 50)
    var = np.hstack((var1, var2, var3))  # (0.1, 10, 0.05)
    varx = np.log10(100 / var)
    file = open('testdata.txt', 'w')
    for i in var:
        # print("正在产生高斯白噪声。。。")
        AWGN_syb = generate_AWGN(i, 9997)
        # print("成功生成高斯白噪声！")
        # print("-----------------------------")

        # print("正在产生信号。。。")
        SIGS1_syb, SIGS2_syb, SIGtmp, SIGS_syb = generate_signal(9997)
        # print("成功生成信号！")
        # print("-----------------------------")

        # print("信号传输中。。。")
        SIGR1_syb = add_AWGN(SIGS1_syb, AWGN_syb)
        SIGR2_syb = add_AWGN(SIGS2_syb, AWGN_syb)
        # print("信号传输成功！")
        # print("-----------------------------")

        # print("接收端正在处理。。。")
        SIGJ1_syb = get_SIGJ(SIGS_syb, SIGR1_syb, sybj1, SIGtmp)
        SIGJ2_syb = get_SIGJ(SIGS_syb, SIGR2_syb, sybj2, SIGtmp)
        # print("接收端处理完毕！")
        # print("-----------------------------")

        BER1 = get_BER(SIGS_syb, SIGJ1_syb)
        BER2 = get_BER(SIGS_syb, SIGJ2_syb)

        txt = str("归一化的信噪比（DB）：") + str(np.log10(100 / i)) + str("DB") + "\n" + str("误比特率：") + "\n" + str("signal Ⅰ：") + str(BER1) + str(
            "   signal Ⅱ(经过信道编码)：") + str(BER2) + "\n" * 2
        file.write(txt)
        print("---------------------------------")
        print("signal Ⅰ：")
        print("信噪比：", np.log10(100 / i), " 误比特率：", BER1)

        print("signal Ⅱ(经过信道编码)：")
        print("信噪比：", np.log10(100 / i), " 误比特率：", BER2)
        print("---------------------------------")

        BERarr1.append(BER1)
        BERarr2.append(BER2)

        # print ("发送端信号集一（符号）:\n", SIGS1_syb)
        # print("噪声（符号）:\n", AWGN_syb)
        # print ("接收端信号集一（符号）:\n", SIGR1_syb)
        # print ("信号集一判决结果（符号）:\n", SIGJ1_syb)
        #
        # print ("发送端信号集二（符号）:\n", SIGS2_syb)
        # print("噪声（符号）:\n", AWGN_syb)
        # print ("接收端信号集二（符号）:\n", SIGR2_syb)
        # print ("信号集二判决结果（符号）:\n", SIGJ2_syb)

        # print("信号一的误比特率为：  ", BER1)
        # print("信号二的误比特率为：  ", BER2)

    file.close()
    plot1 = plb.plot(varx, BERarr1, linestyle='-', color='b', label=u'SET 1')
    plot2 = plb.plot(varx, BERarr2, linestyle='-', color='r', label=u'SET 2')

    plb.title(u'不同信噪比下的误码率')  # give plot a title
    plb.xlabel("Eb/N0")
    plb.ylabel("BER")
    plb.legend()
    plb.show()


if __name__ == '__main__':
    print(u"正在产生。。。")
    plot_BER()
    print(u"完毕！")

    # print("正在产生高斯白噪声。。。")
    # AWGN_syb = generate_AWGN(0.5, 9997)
    # print("成功生成高斯白噪声！")
    # print("-----------------------------")
    # #
    # print("正在产生信号。。。")
    # SIGS1_syb, SIGS2_syb, SIGtmp, SIGS_syb = generate_signal(9997)
    # print("成功生成信号！")
    # print("-----------------------------")
    # #
    # print("信号传输中。。。")
    # SIGR1_syb = add_AWGN(SIGS1_syb, AWGN_syb)
    # SIGR2_syb = add_AWGN(SIGS2_syb, AWGN_syb)
    # plot_SIG(SIGS1_syb, SIGR1_syb, "信号一", 10)#信号一10个周期的发送接收对比
    # plot_SIG(SIGS2_syb, SIGR2_syb, "信号二", 10)#信号二10个周期的发送接收对比
    # # SIGR1_bit = add_AWGN(SIGS1_bit, AWGN_bit)
    # # SIGR2_bit = add_AWGN(SIGS2_bit, AWGN_bit)
    # # print("接收端信号集一（符号）:\n", SIGS1_syb)
    # print("信号传输成功！")
    # print("-----------------------------")
    # #
    # print("接收端正在处理。。。")
    # SIGJ1_syb = get_SIGJ(SIGS_syb, SIGR1_syb, sybj1, SIGtmp)
    # SIGJ2_syb = get_SIGJ(SIGS_syb, SIGR2_syb, sybj2, SIGtmp)
    # print("接收端处理完毕！")
    # print("-----------------------------")
    # #
    # BER1 = get_BER(SIGS_syb, SIGJ1_syb)
    # BER2 = get_BER(SIGS_syb, SIGJ2_syb)
    # #
    # # # print ("发送端信号集一（符号）:\n", SIGS1_syb)
    # # # print("噪声（符号）:\n", AWGN_syb)
    # # # print ("接收端信号集一（符号）:\n", SIGR1_syb)
    # # # print ("信号集一判决结果（符号）:\n", SIGJ1_syb)
    # # #
    # # # print ("发送端信号集二（符号）:\n", SIGS2_syb)
    # # # print("噪声（符号）:\n", AWGN_syb)
    # # # print ("接收端信号集二（符号）:\n", SIGR2_syb)
    # # # print ("信号集二判决结果（符号）:\n", SIGJ2_syb)
    # #
    # print("信号一的误比特率为：  ", BER1)
    # print("信号二的误比特率为：  ", BER2)








