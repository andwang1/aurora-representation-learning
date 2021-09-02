import matplotlib.pyplot as plt

ROOM_H = 10
ROOM_W = 10
discretisation = 20

image ="""
0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1
0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0
0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0
0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0
0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
"""

x = []
y = []

counter_x = 0
counter_y = 0

for entry in image[1::2]:

    if counter_x >= discretisation:
        counter_x = 0
        counter_y += 1

    if entry == "1":
        print("X", counter_x, " INDEX", counter_x * (ROOM_W / discretisation))
        print("Y", counter_y, " INDEX", counter_y * (ROOM_H / discretisation))
        x.append(counter_x * (ROOM_W / discretisation))
        y.append(counter_y * (ROOM_H / discretisation))
    counter_x += 1

print(x)
print(y)

plt.ylim([ROOM_H , 0])
plt.xlim([0, ROOM_W])
plt.scatter(x, y)
plt.show()

#
# y = [i[1] for i in pts]
#
# for i, j in zip(x, y):
#     plt.ylim([ROOM_H , 0])
#     plt.xlim([0, ROOM_W])
#     plt.scatter(i, j)
#     plt.pause(0.8)
#
# # plt.scatter(x, y)
# plt.show()