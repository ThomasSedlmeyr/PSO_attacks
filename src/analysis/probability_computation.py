import pandas as pd
from matplotlib import pyplot as plt

"""
Contains the logic for computing the probability that no target of any classifier is contained in X
"""

def compute_prob_that_no_target_of_any_classifier_is_contained_in_x(num_classifiers, size_d, size_x,
                                                                    sample_with_replacement=True):
    if sample_with_replacement:
        return ((size_d - num_classifiers) / size_d) ** size_x
    else:
        probability = 1.0
        for i in range(num_classifiers):
            probability *= (size_d - size_x - i) / (size_d - i)
        return probability

def compute_probability_table(number_classifiers, size_d, size_x):
    prob_table = []
    for i in range(1, number_classifiers + 1):
        prob_with_replacement = compute_prob_that_no_target_of_any_classifier_is_contained_in_x(i, size_d, size_x, True)
        prob_without_replacement = compute_prob_that_no_target_of_any_classifier_is_contained_in_x(i, size_d, size_x, False)
        prob_table.append({
            'Number of Classifiers': i,
            'Probability with Replacement': prob_with_replacement,
            'Probability without Replacement': prob_without_replacement
        })
    prob_table = pd.DataFrame(prob_table)
    plt.figure(figsize=(12, 6))
    #plt.plot(prob_table['Number of Classifiers'], prob_table['Probability with Replacement'], label='With Replacement',
    #         marker='o')
    plt.plot(prob_table['Number of Classifiers'], prob_table['Probability without Replacement'],
             label='Without Replacement', marker='o')
    plt.xlabel('Number of Classifiers')
    plt.ylabel('Probability')
    plt.title(f'Probability That No Target of Any Classifier Is Contained in X for |X|={size_x} and |D|={size_d}')
    #plt.legend()
    plt.tight_layout()
    plt.grid(True)
    #plt.show()
    plt.savefig(f"probability_table_{size_d}_{size_x}.png", dpi=400)


if __name__ == "__main__":
    num_classifiers = 10
    size_d = 5000
    size_x = 1000
    prob_with_replacement = compute_prob_that_no_target_of_any_classifier_is_contained_in_x(num_classifiers, size_d,
                                                                                            size_x, True)
    prob_without_replacement = compute_prob_that_no_target_of_any_classifier_is_contained_in_x(num_classifiers, size_d,
                                                                                               size_x, False)
    # print the results
    print(f"Probability with replacement: {prob_with_replacement}")
    print(f"Probability without replacement: {prob_without_replacement}")
    compute_probability_table(50, size_d, size_x)
