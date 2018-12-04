# -*- coding: utf-8 -*-
'''
Created on 2017-08-05

@author: naiive
'''


'''
This is the channel encoding comparing that write by naiive.
'''

import copy
import numpy as np
import math
import scipy as sp 
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import logging

plb.mpl.rcParams['font.sans-serif'] = ['SimHei']
plb.mpl.rcParams['axes.unicode_minus'] = False

mul = 100
pos = np.array([+1]*mul)
neg = np.array([-1]*mul)

syb = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
SIGS_syb = []

#SET 1
syb_a1 = np.array([pos, pos, pos, pos, pos, pos])# +1 +1 +1 +1 +1 +1 00
syb_b1 = np.array([neg, neg, neg, pos, pos, pos])# -1 -1 -1 +1 +1 +1 10
syb_c1 = np.array([pos, pos, pos, neg, neg, neg])# +1 +1 +1 -1 -1 -1 01
syb_d1 = np.array([neg, neg, neg, neg, neg, neg])# -1 -1 -1 -1 -1 -1 11
syb1 = [syb_a1, syb_b1, syb_c1, syb_d1]

sybj_a1 = np.array([+mul, +mul, +mul, +mul, +mul, +mul])
sybj_b1 = np.array([-mul, -mul, -mul, +mul, +mul, +mul])
sybj_c1 = np.array([+mul, +mul, +mul, -mul, -mul, -mul])
sybj_d1 = np.array([-mul, -mul, -mul, -mul, -mul, -mul])
sybj1 = [sybj_a1, sybj_b1, sybj_c1, sybj_d1]

SIGS1_bit = []
SIGR1_bit = []
SIGJ1_bit = []
SIGS1_syb = []
SIGR1_syb = []
SIGJ1_syb = []
BER1 = 0

#SET 2
syb_a2 = np.array([pos, pos, neg, neg, neg, neg])# +1 +1 -1 -1 -1 -1 00
syb_b2 = np.array([neg, neg, pos, pos, neg, neg])# -1 -1 +1 +1 -1 -1 10
syb_c2 = np.array([neg, neg, neg, neg, pos, pos])# -1 -1 -1 -1 +1 +1 01
syb_d2 = np.array([pos, pos, pos, pos, pos, pos])# +1 +1 +1 +1 +1 +1 11
syb2 = [syb_a2, syb_b2, syb_c2, syb_d2]

sybj_a2 = np.array([+mul, +mul, -mul, -mul, -mul, -mul])
sybj_b2 = np.array([-mul, -mul, +mul, +mul, -mul, -mul])
sybj_c2 = np.array([-mul, -mul, -mul, -mul, +mul, +mul])
sybj_d2 = np.array([+mul, +mul, +mul, +mul, +mul, +mul])
sybj2 = [sybj_a2, sybj_b2, sybj_c2, sybj_d2]

SIGS2_bit = []
SIGR2_bit = []
SIGJ2_bit = []
SIGS2_syb = []
SIGR2_syb = []
SIGJ2_syb = []
BER2 = 0

SIGtmp = []

#产生num行6列，均值为0，方差为var的AWGN噪声
#以下是我的看法，不知道对不对：
#1.产生两组高斯随机变量X、Y；
#2.Z=deta*sqrt(X.^2+Y.^2)即是符合标准差为deta的瑞利分布随机数.
#https://wenku.baidu.com/view/4bc2a0679b6648d7c1c74613.html

#u=rand(1,n); % 产生（0-1）单位均匀信号
#x=sqrt(2*log2(1./u))*sigma; % 广义均匀分布与单位均匀分布之间的关系
def generate_AWGN(var, num):
    AWGN_syb = []
    for i in range(num):#num行6列,每列都是1行mul列的数组
       AWGN_syb.append(np.random.normal(0, var, size = (6, mul)))
    # AWGN_syb = []
    # for i in range(num):# 生成num行，6列，而每列都是1行mul列的数组
    #     AWGN_syb.append(np.random.rand(6 ,mul))
    # AWGN_syb = np.array(AWGN_syb)
    # AWGN_syb = np.sqrt(-2*var * np.log(1 - AWGN_syb)) # 标准差为deta的瑞利分布
    AWGN_bit = []
    for i in AWGN_syb:
        AWGN_bit.extend(i)
    AWGN_syb = np.array(AWGN_syb)
    AWGN_bit = np.array(AWGN_bit)
    return AWGN_syb, AWGN_bit

#产生num行6列的信号
def generate_signal(num):

    SIGS1_bit = []
    SIGR1_bit = []
    SIGS1_syb = []
    SIGR1_syb = []

    SIGS2_bit = []
    SIGR2_bit = []
    SIGS2_syb = []
    SIGR2_syb = []

    SIGS_syb = []

    SIGtmp = np.zeros((num, 6))

    p = np.random.randint(4, size=num)#0-3的num个随机数
    for i in p:
        SIGS1_syb.append(syb1[i])
        SIGS2_syb.append(syb2[i])
        SIGS_syb.append(syb[i])

    SIGS1_syb = np.array(SIGS1_syb)
    SIGS2_syb = np.array(SIGS2_syb)

    #print ("信号集一（符号）:\n",SIGS1_syb)
    #print ("信号集二（符号）:\n",SIGS2_syb)

    for i in SIGS1_syb:
            SIGS1_bit.extend(i)
    SIGS1_bit = np.array(SIGS1_bit)

    for i in SIGS2_syb:
            SIGS2_bit.extend(i)
    SIGS1_bit = np.array(SIGS1_bit)

    #print ("信号集一（比特）:\n",SIGS1_bit)
    #print ("信号集二（比特）:\n",SIGS2_bit)
    return SIGS1_syb, SIGS1_bit, SIGS2_syb, SIGS2_bit, SIGtmp, SIGS_syb

#添加高斯白噪声，模拟信号传播过程
def add_AWGN(SIGS_, AWGN_):
    SIGR_ = SIGS_ + AWGN_
    return SIGR_

#接收端进行判决
def get_SIGJ(SIGS_syb, SIGR_syb, syb, SIGtmp):
    SIGJ_syb = SIGS_syb.copy()
    # for i in range(len(SIGR_syb)):
    #     for j in range(6):
    #         if SIGR_syb[i][j]>0:
    #             SIGJ_syb[i][j] = +1
    #         else:
    #             SIGJ_syb[i][j] = -1

    SIGJ_syb = np.array(SIGJ_syb)
    for i in range(len(SIGR_syb)):
        for j in range(6):
            # print(np.sum(SIGJ_syb[i][j]))
            # print(SIGJ_syb[i][j])
            SIGtmp[i][j] = np.sum(SIGR_syb[i][j])
    # print(SIGJ_syb)
    for obj in range(len(SIGJ_syb)):
        olen1 = np.sqrt(np.sum(np.square(SIGtmp[obj] - syb[0])))
        olen1f = np.sqrt(np.sum(np.square(SIGtmp[obj] + syb[0])))
        olen2 = np.sqrt(np.sum(np.square(SIGtmp[obj] - syb[1])))
        olen2f = np.sqrt(np.sum(np.square(SIGtmp[obj] + syb[1])))
        olen3 = np.sqrt(np.sum(np.square(SIGtmp[obj] - syb[2])))
        olen3f = np.sqrt(np.sum(np.square(SIGtmp[obj] + syb[2])))
        olen4 = np.sqrt(np.sum(np.square(SIGtmp[obj] - syb[3])))
        olen4f = np.sqrt(np.sum(np.square(SIGtmp[obj] + syb[3])))

        # print(olen1, olen2, olen3, olen4)

        # a = {olen1 : syb[0], olen2 : syb[1], olen3 : syb[2], olen4 : syb[3]}
        a = {olen1: np.array([0, 0]), olen2: np.array([1, 0]), olen3: np.array([0, 1]), olen4: np.array([1, 1])}
        b = {olen1f: syb[0], olen2f: syb[1], olen3f: syb[2], olen4f: syb[3]}
        olen = min([olen1, olen2, olen3, olen4, olen1f, olen2f, olen3f, olen4f])
        olent1 = min([olen1, olen2, olen3, olen4])
        olent2 = min([olen1f, olen2f, olen3f, olen4f])
        SIGJ_syb[obj] = a[olent1]
        #if (olent1 <= olent2):
        #    SIGJ_syb[obj] = a[olent1]
        #else:
        #    SIGJ_syb[obj] = b[olent2]

        # for i in [olen1, olen2, olen3, olen4]:
        #     if i == olenex12:
        #         olen = i


    SIGJ_syb = np.array(SIGJ_syb)
    return SIGJ_syb

#计算误比特率
def get_BER(SIGS_syb, SIGJ_syb):
    # SIGS_syb = np.array(SIGS_syb)
    # for i in range(len(SIGS_syb)):
    #     for j in range(6):
    #         SIGtmp[i][j] = np.sum(SIGS_syb[i][j])

    count = 0
    for i in range(len(SIGJ_syb)):
        for k in ((SIGJ_syb[i] - SIGS_syb[i])):
            if (k != 0):
                count = count + 1

    BER= count / (2*len(SIGS_syb))


    return BER

#绘制误比特率图
def plot_BER():
    BERarr1 = []
    BERarr2 = []
    var = np.arange(1, 101, 10)# (0.1, 10, 0.05)
    varx = 1/var
    for i in var:
        #print("正在产生高斯白噪声。。。")
        AWGN_syb, AWGN_bit = generate_AWGN(i, 9997)
        #print("成功生成高斯白噪声！")
        #print("-----------------------------")

        #print("正在产生信号。。。")
        SIGS1_syb, SIGS1_bit, SIGS2_syb, SIGS2_bit, SIGtmp, SIGS_syb = generate_signal(9997)
        #print("成功生成信号！")
        #print("-----------------------------")

        #print("信号传输中。。。")
        SIGR1_syb = add_AWGN(SIGS1_syb, AWGN_syb)
        SIGR2_syb = add_AWGN(SIGS2_syb, AWGN_syb)
        SIGR1_bit = add_AWGN(SIGS1_bit, AWGN_bit)
        SIGR2_bit = add_AWGN(SIGS2_bit, AWGN_bit)
        #print("信号传输成功！")
        #print("-----------------------------")

        #print("接收端正在处理。。。")
        SIGJ1_syb = get_SIGJ(SIGS_syb, SIGR1_syb, sybj1, SIGtmp)
        SIGJ2_syb = get_SIGJ(SIGS_syb, SIGR2_syb, sybj2, SIGtmp)
        #print("接收端处理完毕！")
        #print("-----------------------------")

        BER1 = get_BER(SIGS_syb, SIGJ1_syb)
        BER2 = get_BER(SIGS_syb, SIGJ2_syb)

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

        #print("信号一的误比特率为：  ", BER1)
        #print("信号二的误比特率为：  ", BER2)

    # plot1 = plb.plot(varx, BERarr1, marker='o', markerfacecolor='r', linestyle='-', color='b', label = u'SET 1')
    # plot2 = plb.plot(varx, BERarr2, marker='o', markerfacecolor='r', linestyle='-', color='r', label = u'SET 2')
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
    # AWGN_syb, AWGN_bit = generate_AWGN(100, 9997)
    # print("成功生成高斯白噪声！")
    # print("-----------------------------")
    # #
    # print("正在产生信号。。。")
    # SIGS1_syb, SIGS1_bit, SIGS2_syb, SIGS2_bit, SIGtmp, SIGS_syb = generate_signal(9997)
    # print("成功生成信号！")
    # print("-----------------------------")
    # #
    # print("信号传输中。。。")
    # SIGR1_syb = add_AWGN(SIGS1_syb, AWGN_syb)
    # SIGR2_syb = add_AWGN(SIGS2_syb, AWGN_syb)
    # SIGR1_bit = add_AWGN(SIGS1_bit, AWGN_bit)
    # SIGR2_bit = add_AWGN(SIGS2_bit, AWGN_bit)
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


    
    


