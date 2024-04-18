import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

multipliers = np.array([24, 70, 41, 21, 60, 47, 82, 87, 80, 35, 73, 89, 100, 90, 17, 77, 83, 85, 79, 55, 12, 27, 52, 15, 30])
hunters = np.array([2, 4, 3, 2, 4, 3, 5, 5, 5, 3, 4, 5, 8, 7, 2, 5, 5, 5, 5, 4, 2, 3, 4, 2, 3])
base_treasure = 7500
cost = 25000

def one_round(multipliers, hunters):
    factor = (100 + np.sum(hunters)) / np.sum(np.sqrt(hunters*multipliers))
    probabilities = 1/100*(factor*np.sqrt(hunters*multipliers) - hunters)
    expected_profit = base_treasure*np.sum((probabilities*multipliers) / (hunters + 100*probabilities))
    return probabilities, expected_profit

def prob_calculator(lam, *data):
    multipliers, hunters, naivety_factor = data
    mult = 1/(100*(naivety_factor)*(naivety_factor)*lam)
    first_addend = (1/2)*(naivety_factor - 1)*np.sum(multipliers)
    second_addend = naivety_factor*lam*np.sum(hunters)
    third_addend = np.sqrt(naivety_factor*lam*multipliers*hunters + (1/4)*(naivety_factor - 1)*(naivety_factor - 1))
    return mult*(first_addend - second_addend + np.sum(third_addend)) - 1

def one_round_naive(multipliers, hunters, naivety_factor=1):
    initial_factor = (100 + np.sum(hunters)) / np.sum(np.sqrt(hunters*multipliers)) # naivety = 1
    lam = fsolve(prob_calculator, initial_factor, args=(multipliers, hunters, naivety_factor))
    assert(lam.size == 1)
    lam = lam[0]
    mult = 1/(100*(naivety_factor)*(naivety_factor)*lam)
    first_addend = (1/2)*(naivety_factor - 1)*multipliers
    second_addend = naivety_factor*lam*hunters
    third_addend = naivety_factor*lam*multipliers*hunters + (1/4)*(naivety_factor - 1)*(naivety_factor - 1)
    probabilities = mult*(first_addend - second_addend + np.sqrt(third_addend))

    expected_profit = base_treasure*np.sum((probabilities*multipliers) / (hunters + 100*naivety_factor*probabilities))

    return probabilities, expected_profit

def two_rounds(multipliers, hunters):
    factor = (200 + np.sum(hunters)) / np.sum(np.sqrt(hunters*multipliers))
    probabilities = 1/100*(factor*np.sqrt(hunters*multipliers) - hunters)
    expected_profit = base_treasure*np.sum((probabilities*multipliers) / (hunters + 100*probabilities)) - cost
    return probabilities, expected_profit

def one_two_rounds(multipliers, hunters):
    probabilities = (1/100)*(np.sqrt((3/10)*hunters*multipliers) - hunters)
    q = 2 - np.sum(probabilities)
    expected_profit = base_treasure*np.sum((probabilities*multipliers) / (hunters + 100*probabilities)) - cost*(1 - q)
    return q, expected_profit

def choose(probs):
    thresholds = np.cumsum(probs)
    seed = np.random.uniform(low=0, high=1, size=1)
    if seed < thresholds[0]:
        return 0
    else:
        selected = (thresholds > seed) & (np.roll(thresholds, 1) < seed)
        assert(np.sum(selected) == 1)
        index = np.argwhere(selected)[0]
        return index[0]

def main():
    probabilities_one, expected_profit_one = one_round(multipliers, hunters)
    print("One round:", probabilities_one)
    print("One round:", expected_profit_one)
    probabilities_one_naive, expected_profit_one_naive = one_round_naive(multipliers, hunters, naivety_factor=2)
    print("One round naive:", probabilities_one_naive)
    print("One round naive:", expected_profit_one_naive)
    probabilities_two, expected_profit_two = one_two_rounds(multipliers, hunters)
    print("Two rounds:", probabilities_two)
    print("Two rounds:", expected_profit_two)


    print("Max indices:", np.argmax(probabilities_one), np.argmax(probabilities_one_naive))
    print("Max values:", np.max(probabilities_one), np.max(probabilities_one_naive))

    plt.figure()
    plt.plot(probabilities_one, color="blue", label="One round")
    plt.plot(probabilities_one_naive, color="red", label="Naive")
    plt.legend()
    plt.show()

    index = choose(probabilities_one_naive)
    print("Multiplier:", multipliers[index])
    print("Hunters:", hunters[index])

    return 0

if __name__ == "__main__":
    main()