import numpy as np
# простая нейронная сеть - девочка


def act(x):
    return 0 if x < 0.5 else 1


def go(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, .3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12]) # матрица 2х3
    weight2 = np.array([-1, 1]) # вектор 1х3

    sum_hidden = np.dot(weight1, x) # вычисляем сумму на ходах нейронов скрытого слоя
    print('Значение сумм на нейронах скрытого слоя: ' + str(sum_hidden))
    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: " + str(out_hidden))
    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print('Выходное значение НС: ' + str(y))
    return y


def result(res):
    return 'нравится' if res == 1 else 'не нравится'


def var1():
    house, rock, attr = 1, 0, 1
    print(result(go(house, rock, attr)))

def var2():
    house, rock, attr = 1, 0, 0
    print(result(go(house, rock, attr)))

if __name__ == '__main__':
    var1()
    var2()
