# Exploratory Data Analysis
"""
Exploratory data analysis is used to summarise the data set and its main
characteristics using visual methods. We have used a variety of
univariate and multivariate visualisation techniques.
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_plots():
    df = pd.read_csv('datafile_edited.csv')
    
    # after creating the df object which contains the dataframe,
    # check that the data is in the correct format.
    df.head(5)     # will print out the first 5 rows
    df.tail(5)     # will print out the last 5 rows
    
    # generate descriptive statistics that summarise the central tendancy, 
    # dispersion and shape of the dataset distribution, excluding NaN values.
    df.describe()
    
    # are there any duplicated rows in the data?
    extra = df[df.duplicated()]
    extra.shape
    
    # separate 'quality' as the target variable and the rest as features
    y = df.quality                      # set 'quality' as target
    X = df.drop('quality', axis=1)      # rest are features
    print(y.shape, X.shape)             # check for correctness
    
    # univariate attribute histogram matrix
    """
    Can view the distribtuion of individual attributes. Can also view any 
    outliers and patterns. 
    """
    sns.set()
    df.hist(figsize=(10,10), xlabelsize=10, ylabelsize=10)
    plt.suptitle('Attribute Histogram Matrix')
    plt.savefig('attribute_histogram_matrix.pdf', bbox_inches='tight')
    plt.show()
    
    
    # multivariate correlation matrix
    """
    Compare attributes and observe how well these features are correlated with
    each other.
    """
    cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
            'pH', 'sulphates', 'alcohol', 'quality']
    cm = np.corrcoef(df[cols].values.T)
    plt.tight_layout
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10,10)) 
    hm = sns.heatmap (cm, cbar=True, annot=True, square=True, fmt='.3f', 
                    annot_kws={'size':9}, yticklabels=cols, xticklabels=cols)
    plt.title('Correlation of Attributes')
    plt.savefig('correlation_matrix.pdf', transparent=True, bbox_inches='tight')
    
    # multivariate scatterplot matrix
    """
    Visualises the pair-wise correlations between the features in this 
    dataset in one place. We can now easily see how the data is distributed 
    and whether it contains any outliers. For example, there is a linear 
    relationship between fixed acidity and density.Furthermore, we can see 
    from the histograms for pH and density that these variables are normally 
    distributed while, the fixed acidity variable is normally distributed as well 
    but contains several outliers.
    """
    sns.set(style="ticks", color_codes=True)
    cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
            'pH', 'sulphates', 'alcohol', 'quality']
    sns.pairplot(df[cols], size = 3.0)
    #plt.suptitle('Multivariate Scatterplot Matrix')
    plt.savefig('multivariate_scatterplot_matrix.pdf', bbox_inches='tight')
    plt.show()
    
    # univariate boxplots (all on one graph) - NOT USED IN PAPER
    pd.options.display.mpl_style = 'default'
    df.boxplot(figsize=(10,5))
    plt.xticks(rotation=90)
    plt.tight_layout
    plt.suptitle('Attribute Box and Whisper Plots')
    plt.savefig('box_and_whisper.pdf', bbox_inches='tight')
    
    # univariate boxplots (all on seperate graphs) 
    """
    Can be used to visualise specific features of individual attributes such as
    mean, std, quartiles and outliers
    """
    fig=plt.figure()
    plt.title('Fixed Acidity')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['fixed acidity'])
    plt.show()
    fig.savefig("boxplot_fixedacidity.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Volatile Acidity')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['volatile acidity'])
    plt.show()
    fig.savefig("boxplot_volatileacidity.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Citric Acid')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['citric acid'])
    plt.show()
    fig.savefig("boxplot_citricacid.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Residual Sugar')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['residual sugar'])
    plt.show()
    fig.savefig("boxplot_residualsugar.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Chlorides')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['chlorides'])
    plt.show()
    fig.savefig("boxplot_chlorides.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Free Sulfur Dioxide')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['free sulfur dioxide'])
    plt.show()
    fig.savefig("boxplot_freesulfurdioxides.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Total Sulfur Dioxide')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['total sulfur dioxide'])
    plt.show()
    fig.savefig("boxplot_totalsulfurdioxides.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Density')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['density'])
    plt.show()
    fig.savefig("boxplot_density.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('pH')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['pH'])
    plt.show()
    fig.savefig("boxplot_pH.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Sulphates')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['sulphates'])
    plt.show()
    fig.savefig("boxplot_sulphates.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Alcohol')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['alcohol'])
    plt.show()
    fig.savefig("boxplot_alcohol.pdf", fmt="pdf")
    
    fig=plt.figure()
    plt.title('Quality')
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(df['quality'])
    plt.show()
    fig.savefig("boxplot_quality.pdf", fmt="pdf")
    
    # univariate density plot example - NOT USED IN PAPER
    df["chlorides"].plot(kind="density",    # Create density plot
                        figsize=(4,4),    # Set figure size
                        xlim= (0,1))      # Limit x axis values