import numpy as np
tmp = np.array([0, 0])
res = np.array([1, 1])
params = np.empty(2)

input_counter = 0
output_counter = 0
temp_index = 0

x = []
y = []

for _ in range(20000):
    tmp = np.array([0, 0])
    res = np.array([1, 1])
    params = np.empty(2)

    input_counter = 0
    output_counter = 0
    temp_index = 0
    gen = np.random.uniform(low=0.0, high=1., size=(12,))
    for i in range(len(gen)):
        if i % 4 == 0 and i > 0:
            res[0] = tmp[0]
            res[1] = tmp[1]
            tmp[0] = 0

        elif i % 4 == 2 and i > 2:
            tmp[1] = 0

        input_counter = i % 2
        output_counter = i % 4
        temp_index = output_counter // 2

        tmp[temp_index] += res[input_counter] * (gen[i] * 2.1 - 0.1)

    params[0] = max(tmp[0], 0.)
    params[1] = max(tmp[1], 0.)

    exponent = len(gen) // 4.01 * 2 + 2

    if len(gen) % 4 == 1:
        params[0] /= pow(2, exponent - 1)
        params[1] /= pow(2, exponent - 2)
    elif len(gen) % 4 == 2:
        params[0] /= pow(2, exponent)
        params[1] /= pow(2, exponent - 2)
    elif len(gen) % 4 == 3:
        params[0] /= pow(2, exponent)
        params[1] /= pow(2, exponent - 1)
    else:
        params[0] /= pow(2, exponent)
        params[1] /= pow(2, exponent)

    # if params[0] < 0:
    #     params[0] += 1
    #
    # if params[1] < 0:
    #     params[1] += 1

    print(params[0])
    assert (params[0] < 1.1)
    print(params[1])
    assert (params[1] < 1.1)

    # params[0] *= 2 * np.pi
    # params[1] *= 0.30
    x.append(params[0])
    y.append(params[1])

import seaborn as sns
import matplotlib.pyplot as plt


sns.distplot(x)
plt.show()
sns.distplot(y)
plt.show()
