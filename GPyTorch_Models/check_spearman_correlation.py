from scipy.stats import spearmanr
x = [1, 2, 3]
x_corr = [1, 4, 7]
corr, p_value = spearmanr(x, x_corr)
print ("Corr: ",corr)
print("p-value: ", p_value)