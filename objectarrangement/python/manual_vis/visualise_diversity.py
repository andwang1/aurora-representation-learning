import matplotlib.pyplot as plt
import numpy as np

OVERLAY = True

ROOM_H = 10
ROOM_W = 10
discretisation = 20

buckets ="00000111111110000000011111000000111110001100000000000000111100111100000000011111000001111100001100000000000001111100000000000000000001111000000000000000000011110000000000000001111100000000000111110000000000001111000000000000111110000000000011111000000000000000111000000000000000000001111000000000000000000011111000000000000000000011111000000000000000000011110000000000000000000000000000000000000000000000011111111000000001111100000011111000110000000000000011110011110000000001111100000111110000110000000000000111110000000000000000000111100000000000000000001111000000000000000111110000000000011111000000000000111100000000000011111000000000001111100000000000000011100000000000000000000111100000000000000000001111100000000000000000001111100000000000000000001111000000000000000000000000000000000000000000"
buckets = reversed(buckets)

rows = []
column = []
counter_x = 0
for i in buckets:
    column.append(int(i))
    counter_x += 1
    if counter_x >= discretisation:
        counter_x = 0
        rows.append(column)
        column = []

# plot colours
plt.ylim([discretisation, 0])
plt.xlim([0, discretisation])
plt.pcolormesh(rows)

# plot grid
plt.grid(which="both")
plt.xticks(range(discretisation), np.arange(0, ROOM_W, ROOM_W / discretisation))
plt.yticks(range(discretisation), np.arange(0, ROOM_H, ROOM_H / discretisation))

plt.title("Diversity Calculation - Single Trajectory")
plt.xlabel("Room X")
plt.ylabel("Room Y")

if OVERLAY:
    trajectory_pts = """
            1
            1
      3.99437
      1.80234
      6.98874
      2.60468
      9.98311
      3.40702
      7.02252
      4.20936
      4.02815
       5.0117
      1.03378
      5.81403
      1.96059
      6.61637
      4.95496
      7.41871
      7.94933
      8.22105
       9.0563
      9.02339
      6.06193
      9.82573
      3.06756
      9.37193
    0.0731892
      8.56959
      2.92118
      7.76725
    """

    # remove first and last empty line
    trajectory_pts = trajectory_pts[1:-1].split("\n")
    # Getting rid of last blank line
    if not trajectory_pts[-1].isnumeric():
        trajectory_pts.pop()
    trajectory_pts = [float(pt.strip()) for pt in trajectory_pts]
    x = [i for i in trajectory_pts[::2]]
    y = [i for i in trajectory_pts[1::2]]

    impact_pts = """
         10
    3.41154
          0
    6.09103
         10
    8.77053
    5.41154
         10
          0
    8.54998
    """

    # remove first and last empty line
    impact_pts = impact_pts[1:-1].split("\n")
    if not impact_pts[-1].isnumeric():
        impact_pts.pop()
    impact_pts = [float(pt.strip()) for pt in impact_pts]
    x_impact = [i for i in impact_pts[::2]]
    y_impact = [i for i in impact_pts[1::2]]

    # need to scale to the discretisation grid size
    x = np.array(x) * discretisation / ROOM_W
    y = np.array(y) * discretisation / ROOM_H
    x_impact = np.array(x_impact) * discretisation / ROOM_W
    y_impact = np.array(y_impact) * discretisation / ROOM_H

    for i, j in zip(x_impact, y_impact):
        plt.scatter(i, j, c="red")
        plt.pause(1)

    for i, j in zip(x, y):
        plt.scatter(i, j, c="blue")
        plt.pause(0.8)

    # Plot lines
    line_x = [pt for pt in x_impact]
    line_y = [pt for pt in y_impact]
    # Add start and end points
    line_x.insert(0, x[0])
    line_y.insert(0, y[0])
    line_x.append(x[-1])
    line_y.append(y[-1])

    plt.plot(line_x, line_y, c="green")
    plt.pause(0.5)

plt.savefig("diversity.pdf")
plt.show()