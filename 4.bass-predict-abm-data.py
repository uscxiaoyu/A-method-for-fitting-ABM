# coding=utf-8
import numpy as np
from bassestimate import BassEstimate, BassForecast


if __name__=='__main__':
    data_set = {'room air conditioners': (np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673,
                                                                  1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425,
                                                            1260, 1236]),
                'ultrasound': (np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography': (np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language': (np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24,
                                                             4.89, 0.25]),
                'accelerated program': (np.arange(1952, 1964), [0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40,
                                                                6.43, 6.15, 1.15])}

    china_set = {'color tv': (np.arange(1997, 2013), [2.6, 1.2, 2.11, 3.79, 3.6, 7.33, 7.18, 5.29, 8.42, 5.68, 6.57,
                                                      5.49, 6.48, 5.42, 10.72, 5.15]),
                 'mobile phone': (np.arange(1997, 2013), [1.7, 1.6, 3.84, 12.36, 14.5, 28.89, 27.18, 21.33, 25.6, 15.88,
                                                          12.3, 6.84, 9.02, 7.82, 16.39, 7.39])}

    S = data_set['room air conditioners'][1]
    bass_fore = BassForecast(S, n=3, b_idx=8)
    res = bass_fore.run()

    print('1步向前预测:', end=' ')
    print('MAD:%.2f  MAPE:%.2f  MSE:%.2f' % res[0])
    print('3步向前预测:', end=' ')
    print('MAD:%.2f  MAPE:%.2f  MSE:%.2f' % res[1])
