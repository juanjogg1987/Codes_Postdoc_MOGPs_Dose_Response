import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

_FOLDER = '/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/PaperPrecOncologyRebuttal_CSVs/'
sel_can = 9
DictFileName = {0:'MOGP_Predict_C0_Train100_m_23.csv',1:'MOGP_Predict_C1_Train100_m_24.csv',
                2:'MOGP_Predict_C2_Train100_m_35.csv',3:'MOGP_Predict_C3_Train100_m_26.csv',
                4:'MOGP_Predict_C4_Train100_m_25.csv',5:'MOGP_Predict_C5_Train100_m_23.csv',
                6:'MOGP_Predict_C6_Train100_m_25.csv',7:'MOGP_Predict_C7_Train100_m_25.csv',
                8:'MOGP_Predict_C8_Train100_m_3.csv',9:'MOGP_Predict_C9_Train100_m_12.csv'}

DictTitleName = {0:'SKCM',1:'SCLC',2:'PAAD',3:'OV',4:'LUAD',5:'HNSC',6:'ESCA',7:'COAD',8:'BRCA',9:'ALL'}

file_name = DictFileName[sel_can]
df = pd.read_csv(_FOLDER+file_name)

import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.colors as mcolors


#mcolors.BASE_COLORS

# Function to create the scatter plot with regression line and text label
def create_scatter_plot(df, title, ax, xlim_scatter, ylim_scatter, xlim_regression, ylim_regression,metric_name='IC50',show_legends=True):
    # Define the amount of jitter
    jitter_amount = 0.03  # Adjust the amount of jitter as needed

    # Add jitter to the x and y coordinates to allow visualising the data that overlaps!!
    df[metric_name + '_MOGP_jittered'] = df[metric_name + '_MOGP'] + np.random.uniform(-jitter_amount, jitter_amount,len(df))
    df[metric_name + '_s4_jittered'] = df[metric_name + '_s4'] + np.random.uniform(-jitter_amount, jitter_amount,len(df))

    # Set the x-axis and y-axis limits for the scatter plot
    ax.set_xlim(xlim_scatter)
    ax.set_ylim(ylim_scatter)

    if show_legends:
        sns.scatterplot(data=df, x=metric_name + '_MOGP_jittered', y=metric_name + '_s4_jittered', hue='DRUG_ID',
                        palette=['#009E73','#CC79A7','#F0E442','#0072B2','#B22222','#8C8C8C','#1A1A1A','#56B4E9','#E69F00','#800080'],
                        hue_order=[1012,1021,1036,1053,1058,1059,1061,1149,1372,1373], alpha=0.65, ax=ax,legend = "full")
    else:
        sns.scatterplot(data=df, x=metric_name + '_MOGP_jittered', y=metric_name + '_s4_jittered',palette=['#009E73', '#CC79A7'], alpha=0.55, ax=ax)

    # Calculate and plot the regression line manually
    x_reg = df[metric_name + '_MOGP']
    y_reg = df[metric_name + '_s4']
    slope, intercept = np.polyfit(x_reg, y_reg, 1)  # Calculate the slope and intercept
    x_range = np.linspace(xlim_regression[0], xlim_regression[1],100)  # Define the x-axis range for the regression line
    y_range = slope * x_range + intercept  # Calculate the corresponding y-values
    ax.plot(x_range, y_range, color='darkgrey', lw=1.8)  # Plot the extended regression line

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel('Predicted ' + metric_name, fontsize=15)
    ax.set_ylabel('Observed ' + metric_name, fontsize=15)

    #ax.get_legend().remove()
    #ax.legend(loc='upper center',ncol=10)
    if show_legends:
        ax.legend(title='DrugID',bbox_to_anchor=(0.0, 0.56),loc='center left',prop = {'size': 8.5})

    # Calculate Pearson correlation coefficient and p-value
    r, p_value = pearsonr(df[metric_name + '_MOGP'], df[metric_name + '_s4'])

    if metric_name == 'Emax':
        ax.annotate(f'r = {r:.3f}', xy=(0.08, 1.0), color='black', size=12.5, va='center')
    elif metric_name == 'AUC':
        ax.annotate(f'r = {r:.3f}', xy=(0.08, 1.0), color='black', size=12.5, va='center')
    elif metric_name == 'IC50':
        ax.annotate(f'r = {r:.3f}', xy=(0.08, 1.57), color='black', size=12.5, va='center')#

# Create subplots for both plots

fig, axes = plt.subplots(3, 1, figsize=(5.5, 20))
#axes[0].set_title(f'GDSC2')
#fig, axes = plt.subplots(1, 2, figsize=(10, 4))
show_legend = True
create_scatter_plot(df, '', axes[0],
                   xlim_scatter=(0.05, 1.05),  # Scatter plot axis limits
                        ylim_scatter=(0.05, 1.05),  # Scatter plot axis limits
                        xlim_regression=(0.15, 1.0),   # Regression line axis limits
                        ylim_regression=(0.15, 1.0),
                    metric_name = 'Emax',
                    show_legends=show_legend
                   )
create_scatter_plot(df, '', axes[1],
                   xlim_scatter=(0.05, 1.65),  # Scatter plot axis limits
                        ylim_scatter=(0.05, 1.65),  # Scatter plot axis limits
                        xlim_regression=(0.2, 1.6),   # Regression line axis limits
                        ylim_regression=(0.2, 1.6),
                    metric_name = 'IC50',
                    show_legends=show_legend
                   )
create_scatter_plot(df, '', axes[2],
                   xlim_scatter=(0.05, 1.05),  # Scatter plot axis limits
                        ylim_scatter=(0.05, 1.05),  # Scatter plot axis limits
                        xlim_regression=(0.15, 1.03),   # Regression line axis limits
                        ylim_regression=(0.15, 1.03),
                    metric_name = 'AUC',
                    show_legends=show_legend
                   )

axes[0].set_title(f'MOGP performance on Test data for {DictTitleName[sel_can]}',fontsize=15)