import numpy as np
import pandas as pd

# https://www.machinelearningplus.com/statistics/t-test-students-understanding-the-math-and-how-it-works/
def t_test(scores_A, scores_B, iter, p):

    accuracy_A = np.mean(scores_A)
    accuracy_B = np.mean(scores_B)
    delta_means = accuracy_A - accuracy_B
    delta_sum = 0

    #compute the variance of the difference between the two models
    for i in range(0,iter):
        delta_i = scores_A[i] - scores_B[i]
        delta_sum += (delta_i - delta_means)*(delta_i - delta_means)
    variance = delta_sum/iter

    #t-statistic
    t_statistic = delta_means/np.sqrt(variance/iter)

    # degrees of freedom
    df = iter-1
    t_table = pd.read_csv("./document/t_distribution_table.csv")
    c = float(t_table.loc[df, str(round(p/2, 3))])

    if t_statistic > c or t_statistic < -c: #if t_statistic
        response = 'We can reject the null-hypothesis that both models perform ' \
                   'equally well on this dataset. We may conclude that the two ' \
                   'algorithms are significantly different.'
    else:
        response = 'We cannot reject the null hypothesis and may conclude that ' \
                   'the performance of the two algorithms is not significantly ' \
                   'different.'

    return t_statistic, c, response




