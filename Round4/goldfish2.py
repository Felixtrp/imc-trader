import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def profit(a, x, y):
    part1 = (100-x)*x*x
    if y >= a:
        part2 = (100-y)*(x+y)*(y-x)
    else:
        part2 = (100-y)*(x+y)*(y-x)*(100-a)/(100-y)

    return (1/10000)*(part1 + part2)

def main():
    
    results = np.zeros(shape=(101, 101, 101))
    for a in np.arange(101):
        for y in np.arange(101):
            for x in np.arange(y+1):
                results[a, x, y] = profit(a, x, y)

    x_values = np.zeros(shape=101)
    y_values = np.zeros(shape=101)
    max_profits = np.zeros(shape=101)

    for i in range(101):
        x_values[i], y_values[i] = np.unravel_index(np.argmax(results[i]), shape=results[i].shape)
        max_profits[i] = results[i].max()

    plt.figure()
    plt.plot(max_profits, label="Profits")
    plt.plot(x_values, label="Low bid")
    plt.plot(y_values, label="High bid")
    plt.plot(np.arange(101), label="Average bid")
    plt.legend()
    plt.show()

    for i in range(101):
        print("Average", i, ":", max_profits[i], results[i, 52, 78])

    print(np.argmax(results[0, :, 100]))
    print(results[0, 50, 100])

    return 0

if __name__ == "__main__":
    main()