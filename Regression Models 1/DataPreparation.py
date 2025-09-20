import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import f_oneway, chi2_contingency
from sklearn.feature_selection import chi2, f_classif
import math 
import re 
import numbers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway



# clean and readable column names
def clean_columns(data, change_signs = False):
    """

    This function standardizes column names by removing any leading or trailing whitespace, ensuring consistent and clean naming for further data processing.

    Parameters: 
    -----------
    data : Pandas DataFrame.

    change_signs : if True, then all non-alphanumeric characters will be converted to underscore. If False, then non-alphanumeric characters will be untouched.

    """
    columns = data.columns.str.strip()
    columns = [re.sub(r'([a-z])([A-Z])',r'\1_\2', col).lower() for col in columns]
    
    if change_signs:
        columns = [re.sub(r'[^A-Za-z0-9_]', r'_',col) for col in columns]
     
    data.columns = columns
    return data 




# clean and readable string values 
def clean_values(data,column,change_signs = False, uppercase = False, lowercase = False):
    """ 
    This function cleans the values inside the given columns.

    Parameters: 
    -----------
    data : pandas DataFrame

    column : column that you want to apply cleaning 

    change_signs : True if you want to change non-alphanumeric signs to underscore (yes? -> yes_)

    uppercase : if True, then all values wil be uppercase

    lowercase: if False, then all values will be lowercase


    """

    # remove spaces before and after each string
    data[column] = data[column].str.strip().str.replace(r'[\s]+', r'_',regex=True)


    # if user wants to convert non-alphanumeric to underscore
    if change_signs:
        data[column] = data[column].str.replace(r'[^a-zA-Z0-9_]',r'_',regex=True)
    
    # if user wants to convert characters to uppercase
    if uppercase:
        data[column] = data[column].str.upper()

    # if user wants to convert characters to lowercase
    if lowercase:
        data[column] = data[column].str.lower()

    return data          



# show missing values as table and heatmap

def check_missing(data, columns='all', visual = False, width = 10, height = 6):
    
    """
    This function checks the missing values in the given column.

    Parameters: 
    -----------
    data : pandas DataFrame

    columns : a list of columns in which you want to find missing values.

    visual : True, if you want to visualize missing values as seaborn heatmap.

    width : Width of the figure (default is 10)

    Height : Height of the figure (default is 6)

    Returns a table showing the number of missing values and the percentage of total.

    """
    
    if columns == 'all':
        missing_table = data.isnull().sum().to_frame(name='Missing')

    else:
        if isinstance(columns,str):
            columns = [columns]
        missing_table = data[columns].isnull().sum().to_frame(name='Missing')
    
    missing_table['%_of_total'] = round((missing_table['Missing'] / data.shape[0]) * 100,2)

    if visual: 
        plt.figure(figsize=(width,height))
        if columns == 'all':
            sns.heatmap(data.isnull(), cbar=False, cmap=['black', 'white'])
            plt.title('Heatmap of missing columns')
            plt.xticks(rotation=45)
        else: 
            data = data[columns]
            sns.heatmap(data.isnull(), cbar=False, cmap=['black', 'white'] )
            plt.title('Heatmap of missing columns')
        plt.show()
    return missing_table



# handle missing values 
def handle_missing(data, columns, dtype, by):
    """
    This function handles missing values by different methods. 

    Parameters:
    -----------
    data : Pandas dataframe.

    columns : Specified columns in which you want to handle missing values

    dtype : data type of columns. Because, data type differs between numerical and categorical columns.

    by : If dtype is numeric, then select 'mean', 'median', 'mode' or 'constant number'. If dtype is categoric (object), then select either 'mode' or 'constant string'    
    
    Returns dataframe with missing values are handled in the specified columns.
    """

    # if one string is entered as argument, then make it a list
    if isinstance(columns,str):
        columns=[columns]

    
    # if data type is numeric, then either mean, median, mode or constant 
    if dtype == 'number':
        # if by is string, then it shows the filling method: mean, median or mode
        if isinstance(by, str):
            for col in columns:
                if by == 'mean':
                    data[col] = data[col].fillna(data[col].mean())
                elif by == 'median':
                    data[col] = data[col].fillna(data[col].median())
                elif by == 'mode':
                    data[col] = data[col].fillna(data[col].mode()[0])
                else:
                    raise ValueError("Invalid fill method for numeric column")
        elif isinstance(by, numbers.Number):
            data[columns] = data[columns].fillna(by)
    
    
    # if data type if object (categoric), then either mode or constant string 
    if dtype == 'object': 
        if isinstance(by, numbers.Number):
            raise ValueError("Can not assign number to categorical variables! Write either mode or \
                             constant string!")
        
        for col in columns: 
            if by == 'mode':
                data[col] = data[col].fillna(data[col].mode()[0])
            else: 
                data[col] = data[col].fillna(by)            
            
    return data 


# check duplicates
def check_duplicates(data, visual=False, handle=False):
    """
    This function checks the duplicate rows. 

    Parameters:
    -----------

    data : Pandas dataframe

    visual : If True, then shows the duplicated rows as a dataframe

    handle : If True, then removes the duplicated rows by keeping only the first row
    
    """
    print("number of duplicates: ", data.duplicated().sum())

    if visual:
        return data[data.duplicated(keep=False)==True]
    
    if handle:
        data = data.drop_duplicates(keep='first')
        return data 

    if visual and handle:
        raise ValueError("Set either visual=True or handle=True, not both")
    
    return None 
    




# show outliers/extremes as table and box-plots
def check_outliers(data, columns, table=True, boxplot=False, width=10, height=7):
    """
    This function checks the outliers in the specified columns of the given dataframe.
    
    Parameters: 
    -----------
    
    data : pandas datafame

    columns : columns that outliers may exist

    table : if True, function will return the table showing the number of outliers in the columns

    boxplot : if True, outliers will be shown as a boxplot.  
    """

    if isinstance(columns,str):
        raise ValueError("Outliers can not be checked for object dtype columns. Please, \
                         include only numeric columns!")
    
    # generate copy dataframe from the given data and columns
    temp_df = data[columns].copy()

    # apply standard scaler to convert all variables to z-score
    scaler = StandardScaler()
    temp_df[columns] = scaler.fit_transform(temp_df[columns])

    # now for each column, print the number of outliers
    # the outliers are those who has the value more than 3 or less than -3
    outliers = dict()
    for col in columns: 
        num_of_outliers = temp_df[(temp_df[col]>3) | (temp_df[col]<-3)].shape[0]
        perc_of_total = round(temp_df[(temp_df[col]>3) | (temp_df[col]<-3)].shape[0] / temp_df.shape[0] * 100,2)
        outliers[col] = [num_of_outliers, perc_of_total]

    outliers_df = pd.DataFrame(outliers, index=['num_of_outliers', 'perc_of_total']).T
    outliers_df['num_of_outliers'] =  outliers_df['num_of_outliers'].astype(int)

    if boxplot: 
        cols=3
        rows = int(np.ceil(len(columns)/3))
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height))
        axes = axes.flatten()
        for i,col in enumerate(columns): 
            sns.boxplot(data=data, x=col, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')

        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])



        plt.tight_layout()
        plt.show()
    else:
        return outliers_df







# handle outliers: coerce , drop
def handle_outliers(data, columns, by='coerce'):
    """
    Handle outliers in specified numeric columns of a pandas DataFrame using either capping or removal.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing numeric columns to process.

    columns : str or list of str
        Column name or list of numeric columns in which to handle outliers.

    by : str, default 'coerce'
        Method to handle outliers:
        - 'coerce' : Caps values outside the interquartile range (IQR) at the lower and upper bounds.
        - 'drop' : Removes rows where values fall outside the IQR bounds.

    Returns
    -------
    pandas.DataFrame
        DataFrame with outliers either capped or removed according to the specified method.

    Notes
    -----
    - Outliers are defined as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    - Works only on numeric columns.
    - If a single column is passed as a string, it is converted to a list internally.
    """

    if isinstance(columns,str):
        columns = [columns]

    if by == 'coerce':
        for col in columns: 
            Q1 = np.percentile(data[col],25)
            Q3 = np.percentile(data[col],75)
            IQR = Q3 - Q1 
            lower_bound = Q1 - 1.5*IQR 
            upper_bound = Q3 + 1.5*IQR 

            data[col] = np.clip(data[col], a_min=lower_bound, a_max=upper_bound)
        
    elif by == 'drop':
        mask = np.ones(len(data), dtype=bool)
        for col in columns:
            if data[col].dropna().empty:
                continue  # skip empty columns safely
            Q1 = np.percentile(data[col],25)
            Q3 = np.percentile(data[col],75)
            IQR = Q3 - Q1 
            lower_bound = Q1 - 1.5*IQR 
            upper_bound = Q3 + 1.5*IQR 

            mask &= (data[col] > lower_bound) & (data[col] <= upper_bound)
        
        data = data[mask].reset_index(drop=True)

    else: 
        raise ValueError("Wrong 'by' method. Use either 'drop' or 'coerce' method.")
    
    return data 



# show distribution of numeric features 
def check_distribution(data, columns,hue=None, histogram=False, boxplot=False, violinplot=False, width=10,
                       height=6, bins='auto', kde=False):

    """
    Visualize the distribution of numeric features in a pandas DataFrame using histogram, boxplot, or violin plot.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing numeric columns to visualize.

    columns : str or list of str
        Column name or list of numeric columns to visualize.

    hue : str, optional
        Column name for grouping data by a categorical variable (default is None).

    histogram : bool, default False
        If True, plots histogram(s) for the specified columns.

    boxplot : bool, default False
        If True, plots boxplot(s) for the specified columns.

    violinplot : bool, default False
        If True, plots violin plot(s) for the specified columns.

    width : int, default 10
        Width of the figure for each plot type.

    height : int, default 6
        Height of the figure for each plot type.

    bins : int, str, or sequence, default 'auto'
        Number of bins or binning strategy for histograms (passed to `sns.histplot`).

    kde : bool, default False
        If True and `histogram=True`, overlays a kernel density estimate on the histogram.

    Returns
    -------
    None
        Displays the selected plots inline.

    Notes
    -----
    - For multiple columns, subplots are arranged with 3 columns per row.
    - Any extra subplot axes are removed for cleaner visualization.
    - Use `hue` to separate the distributions by a categorical variable.
    """
    
    if isinstance(columns, str):
        columns=[columns]
    
    if histogram:
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/3), ncols=3, figsize=(width,height))
        axes = axes.flatten()
        for i,col in enumerate(columns):
            sns.histplot(data=data, x=col, hue=hue,bins=bins, kde=kde, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')

        for i in range(len(columns),len(axes)):
            fig.delaxes(axes[i])
        plt.show()

    if boxplot:
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/3), ncols=3, figsize=(width,height))
        axes = axes.flatten()
        for i,col in enumerate(columns):
            sns.boxplot(data=data,x = col , hue=hue, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')

        for i in range(len(columns),len(axes)):
            fig.delaxes(axes[i])
        plt.show()


    if violinplot:
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/3), ncols=3, figsize=(width,height))
        axes = axes.flatten()
        for i,col in enumerate(columns):
            sns.violinplot(data=data,x = col , hue=hue, ax=axes[i])
            axes[i].set_title(f'Violinplot of {col}')

        for i in range(len(columns),len(axes)):
            fig.delaxes(axes[i])
        plt.show()




# scale numeric features, method=minmax, standard, robust
def scale_variables(data,method ,columns=None, overwrite = False):
    """
    Scale numeric features in a pandas DataFrame using Standard, MinMax, or Robust scaling.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe containing numeric columns to scale.

    method : str
        Scaling method to use. Options are:
        - 'standard' : StandardScaler (zero mean, unit variance)
        - 'minmax'   : MinMaxScaler (scales values to [0, 1])
        - 'robust'   : RobustScaler (scales using median and interquartile range, robust to outliers)

    columns : str or list of str, optional
        Column name or list of numeric column names to scale. If None, all numeric columns are scaled.

    overwrite : bool, default False
        If True, original numeric columns are replaced with their scaled versions.
        If False, new columns are added with suffix '_scaled'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with scaled numeric columns added or replaced according to `overwrite`.

    Raises
    ------
    ValueError
        - If no numeric columns are provided for scaling.
        - If `method` is not one of the supported options ('standard', 'minmax', 'robust').

    Notes
    -----
    - When `overwrite=False`, columns that already end with '_scaled' are skipped to prevent double scaling.
    - This function uses sklearn scalers internally and returns the DataFrame with scaled values.
    """

    if isinstance(columns, str):
        columns = [columns]


    if columns is None:
        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    else: 
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(data[c])]


    if not numeric_cols:
        raise ValueError('No numeric columns is provided for scaling')
    
    if overwrite:
        if method.lower() == 'standard':
            scaler = StandardScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        elif method.lower() == 'minmax':
            scaler = MinMaxScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        elif method.lower() == 'robust':
            scaler = RobustScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        else: 
            raise ValueError('Wrong method selected! Select one of these: standard, minmax, robust!')
        
    elif overwrite==False:
        if method.lower() == 'standard':
            for col in numeric_cols:
                if len(re.findall(r'_scaled$', col))!=0:
                    continue 
                scaler = StandardScaler()
                data[col + '_scaled'] = scaler.fit_transform(data[[col]])

        
        elif method.lower() == 'minmax':
            for col in numeric_cols:
                if len(re.findall(r'_scaled$', col))!=0:
                    continue 
                scaler = MinMaxScaler()
                data[col + '_scaled'] = scaler.fit_transform(data[[col]])
        
        elif method.lower() == 'robust':
            for col in numeric_cols:
                if len(re.findall(r'_scaled$', col))!=0:
                    continue 
                scaler = RobustScaler()
                data[col + '_scaled'] = scaler.fit_transform(data[[col]])
        else: 
            raise ValueError('Wrong method selected! Select one of these: standard, minmax, robust!')

    return data 





# encode categoric features, method=onehot, dummy, label, ordinalencoder
def encode_variables(data, columns, method, order_dict = None): 
    """
    Encode categorical features in a pandas DataFrame using different encoding methods.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe containing the categorical columns to encode.

    columns : str or list of str
        Column name or list of column names to encode.

    method : str
        Encoding method. Options are:
        - 'onehotencoder' : Uses sklearn's OneHotEncoder to create binary columns for each category.
        - 'dummy'         : Uses pandas get_dummies, dropping the first category to avoid multicollinearity.
        - 'ordinalencoder' : Encodes categories as integers based on a specified order.

    order_dict : dict, optional
        Required only for ordinal encoding. Dictionary mapping column names to ordered category lists.
        Example: {'education': ['no degree', 'bachelor', 'master', 'phd']}

    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded categorical columns added. Original columns are retained unless using 'dummy' encoding (where original columns are replaced).

    Raises
    ------
    ValueError
        - If 'ordinalencoder' is selected and `order_dict` is not provided.
        - If `method` is not one of the valid options.

    Notes
    -----
    - Columns that already contain '_encoded' in their name are skipped to prevent double encoding.
    - For OneHotEncoder, new column names are created in the format '<original_column>_<category>'.
    """
   
    if isinstance(columns, str):
        columns = [columns]
    method = method.lower()
    if method == 'onehotencoder':
        encoder = OneHotEncoder(sparse_output=False)
        for col in columns: 
            if len(re.findall(r'_encoded', col))!=0:
                continue 
            encoded_array = encoder.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(encoded_array, columns=[f'{col}_{cat}' for cat in encoder.categories_[0]])
            data=pd.concat([data,encoded_df],axis=1)
    
    elif method == 'dummy':
        data = pd.get_dummies(data, columns=columns, drop_first=True, dtype=int)
        
    elif method == "ordinalencoder": # order_dictionary {education: no degree, bachelor, master, phd,
                                                    # age: young, middle, old}

        if order_dict is None: 
            raise ValueError('For ordinal encoding, provide the order_dict = {col: [order_list]}')
        for (col, order) in order_dict.items():
            if len(re.findall(r'_encoded', col))!=0:
                continue 
            data[col + '_encoded'] = data[col].map({val:i for i,val in enumerate (order)})

    else: 
        raise ValueError('Invalid method! Choose: OneHotEncoder, dummy, OrdinalEncoder')

    return data 

        



# correlataion between numeric features, pearson spearman
def correlation(data, columns=None, target=None, visual=False, all_cols = False, method = 'Pearson',
                width = 10, height = 7):
    """
    Compute and optionally visualize the correlation between numeric features in a dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe containing numeric features.

    columns : list of str, optional
        List of columns to include in the correlation calculation. If None, all numeric columns are used.

    target : str, optional
        Target column for which correlation with other features is calculated. 
        Required if `all_cols` is False.

    visual : bool, default=False
        If True, displays a heatmap of the correlation values using seaborn.

    all_cols : bool, default=False
        If True, correlation is computed for all numeric columns.
        If False, correlation is computed only with respect to the target column.

    method : str, default='Pearson'
        Correlation method. Currently only Pearson is supported (for Spearman, additional code is required).

    width : int, default=10
        Width of the figure if `visual=True`.

    height : int, default=7
        Height of the figure if `visual=True`.

    Returns
    -------
    pandas.DataFrame
        Correlation table:
        - If `all_cols=False` and `visual=False`, returns correlations of numeric features with the target column.
        - If `all_cols=True` and `visual=False`, returns the full correlation matrix.
        - Returns None if `visual=True` (plots only).

    Raises
    ------
    ValueError
        If `all_cols=False` and `target` is not specified.
        If invalid argument combination is provided.
    
    Notes
    -----
    - The function filters numeric columns automatically.
    - Heatmap annotations are enabled if `visual=True`.
    """

    if columns is not None: 
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
    if target is not None: 
        numeric_cols.append(target)
    if columns is None: 
        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]     
        columns = numeric_cols   
    corr_table = data[numeric_cols].corr()

    if (all_cols == False) & (visual == False): 
        if target is None: 
            raise ValueError("Specify target if you don't want correlation for all columns!")
        corr_table = corr_table[[target]].sort_values(by=target, ascending=False)
        corr_table = corr_table.T
        return corr_table
    

    
    elif (all_cols == False) & (visual == True): 
        if target is None: 
            raise ValueError("Specify target if you don't want correlation for all columns!")
        plt.figure(figsize=(width, height))
        corr_table = corr_table[[target]].sort_values(by=target, ascending=False)
        corr_table = corr_table.T
        sns.heatmap(corr_table, cbar=False,annot=True)
        plt.show()

    
    elif all_cols==True and  visual==True: 
        plt.figure(figsize=(width, height))
        sns.heatmap(corr_table, cbar=False,annot=True)
        plt.show()

    elif all_cols==True and visual == False: 
        return corr_table
    else: 
        raise ValueError('Invalid Argument!')




# relationship between numeric features and categoric target. ANOVA test 
def anova(data, columns, target, visual=False, color='#89a832', annot = False, 
          fontcolor='#157d59', width=10, height=7):
    """
    Perform one-way ANOVA to evaluate the relationship between numeric features and 
    a categorical target, and optionally visualize feature importance.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing features and target.

    columns : list of str
        List of numeric feature columns to evaluate.

    target : str
        Name of the categorical target column.

    visual : bool, default=False
        If True, displays a horizontal bar chart of F-statistics for the features.

    color : str, default='#89a832'
        Color of the bars in the plot.

    annot : bool, default=False
        If True, annotates the bars with F-statistic, p-value, and significance.

    fontcolor : str, default='#157d59'
        Color of the annotation text.

    width : int, default=10
        Width of the figure for visualization.

    height : int, default=7
        Height of the figure for visualization.

    Returns
    -------
    anova_df_sorted : pandas.DataFrame
        DataFrame containing F-statistics, p-values, and significance for each feature,
        sorted by descending F-statistic.

    Notes
    -----
    - Features with insufficient variance across target groups are assigned NaN values.
    - Significance is determined at a p-value threshold of 0.005.
    """

    target_values = data[target].dropna().unique()
    num_of_groups = len(target_values)

    anova_results = {}

    for col in columns: 
        groups = []
        
        for i in range(num_of_groups):
            groups.append(data[data[target]==target_values[i]][col])

        if any(len(np.unique(g))<2 for g in groups):
            f_stat, p_value = np.nan, np.nan 
        else: 
            f_stat, p_value = f_oneway(*groups) 
        
        if p_value < 0.005:
            anova_results[col] = [f_stat, p_value,'sign.']
        else:
            anova_results[col] = [f_stat, p_value,'insign.']

    anova_df = pd.DataFrame(anova_results, index = ['f_stat', 'p_value','significance'])
    anova_df = anova_df.T.sort_values(by='f_stat',ascending=True)
    

    anova_df_sorted = anova_df.sort_values(by='f_stat',ascending=False)

    if visual==False: 
        return anova_df_sorted
    
    else:
        plt.figure(figsize=(width, height))
        bars = plt.barh(anova_df.index,anova_df['f_stat'], color=color)
        plt.title("Feature Importance", fontsize=14, fontweight="bold")

    
    if annot:
        for i, bar in enumerate(bars):
            bar_width = bar.get_width()
            feature = anova_df.index[i]
            f = anova_df.loc[feature,'f_stat']
            p = anova_df.loc[feature,'p_value']
            sign = anova_df.loc[feature,'significance']

        # Dynamic text position
            if bar_width > 0.1 * anova_df['f_stat'].max():  # long enough bar → text inside
                x_text = bar_width * 0.5
                ha = 'center'
            else:  # short bar → text outside left
                x_text = bar_width + 0.01 * anova_df['f_stat'].max()
                ha = 'left'
            
            plt.text(x_text, 
                    bar.get_y() + bar.get_height()/2,
                    f'F: {f:.3f}\np: {p:.4f} \n{sign}',
                    ha=ha, va='center', color=fontcolor, fontsize=12, fontweight='bold')
    plt.show()



# relationships between categoric features. Chi square
def chi2(data, columns, target, visual=False,color= '#89a832',annot=False,fontcolor='#157d59', 
        width=10, height=7): 
    """
    Computes Chi-square statistics and Cramér's V to evaluate the strength of association 
    between categorical features and a categorical target.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing the categorical variables.
        
    columns : list of str
        List of categorical feature column names to analyze.
        
    target : str
        Name of the categorical target column.
        
    visual : bool, default=False
        If True, plots a horizontal bar chart of Cramér's V values for the features.
        
    color : str, default='#89a832'
        Color of the bars in the plot.
        
    annot : bool, default=False
        If True, annotates the bars with Cramér's V and p-value.
        
    fontcolor : str, default='#157d59'
        Color of the annotation text.
        
    width : int, default=10
        Width of the figure for visualization.
        
    height : int, default=7
        Height of the figure for visualization.

    Returns
    -------
    chi2_df : pandas.DataFrame
        DataFrame containing the Chi-square statistic, p-value, and Cramér's V for each feature.
        Sorted by ascending Cramér's V.

    Notes
    -----
    - Cramér's V is a measure of association between two categorical variables and ranges from 0 (no association) to 1 (perfect association).
    - p-value indicates the statistical significance of the Chi-square test.
    """
    

    if isinstance(target,str)==False: 
        target = target[0]

    chi2_results = {}
    for col in columns: 
        cross_tab = pd.crosstab(data[col], data[target])
        chi2, pvalue, dof, _  = chi2_contingency(cross_tab)

        # calculate cramers v to find stenegth of relationship
        n = cross_tab.sum().sum()
        nrows, ncols = cross_tab.shape 
        cramers_v = np.sqrt(chi2/(n * min(nrows, ncols)-1))

        chi2_results[col] = [chi2, pvalue, cramers_v]

    chi2_df = pd.DataFrame(chi2_results, index=['chi2','pvalue', 'cramers_v' ])
    chi2_df = chi2_df.T.sort_values(by='cramers_v', ascending=True)

    
    if visual:
        plt.figure(figsize=(width, height))
        bars = plt.barh(chi2_df.index, chi2_df['cramers_v'], color=color)
        plt.title("Feature Importance", fontsize=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=14)

    if annot:
        for i, bar in enumerate(bars):
            width = bar.get_width()
            feature = chi2_df.index[i]
            pvalue = chi2_df.loc[feature,'pvalue']
            cr_v = chi2_df.loc[feature, 'cramers_v']
            plt.text(width*0.5, bar.get_y()+bar.get_height()/2,
                    f"V: {cr_v:.4f}, P: {pvalue:.3f}",
                    ha='center', va='center', color=fontcolor, fontsize=10, fontweight='bold')

        plt.show()

    return chi2_df




def feature_importance(data, categoric_cols, numeric_target, method='anova', visual=False, color='#89a832', annot=False, 
                       fontcolor='#157d59', width=10, height=7):
    """
    Evaluate feature importance using ANOVA or Chi-square tests and optionally visualize the results.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing features and target.

    columns : list of str
        List of feature column names to evaluate.

    target : str
        Name of the target column.

    method : str, default='anova'
        Method to use for feature importance:
        - 'anova' : For numeric features with a categorical target.

        - 'chi2'  : For categorical features with a categorical target.

    visual : bool, default=False
        If True, displays a horizontal bar chart of feature importance.
    color : str, default='#89a832'
        Color of the bars in the plot.
    annot : bool, default=False
        If True, annotates the bars with relevant statistics.
    fontcolor : str, default='#157d59'
        Color of the annotation text.
    width : int, default=10
        Width of the figure for visualization.
    height : int, default=7
        Height of the figure for visualization.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing feature importance statistics, sorted by importance.
        - For 'anova', returns F-statistics and p-values.
        - For 'chi2', returns Chi-square statistics, p-values, and Cramér's V.
   """

    if isinstance(categoric_cols,str):
        categoric_cols = [categoric_cols]
        
    # columns are categorical variables 
    anova_results = {}
    for col in categoric_cols:
        # col: Gender : ['male', 'female', 'other']

        # group_#1 -> groupby male
        # group_#2 -> groupby female
        # group_#3 -> groupby other

        num_of_groups = len(data[col].dropna().unique())  # num_of_groups = 3
        values = data[col].dropna().unique() # [male, female, othere]
        groups = []

        for i in range(num_of_groups):
            # i = 0
            group = data[data[col] == values[i]][numeric_target]
            groups.append(group)
        
        # groups = [[array_of_male], [array_of_female], [array_of_other]] -> array of 3 groups

        # find anova result for specific categorical column
        # check each group to have at least 2 elements
        if any(len(g)<2 for g in groups):
            anova_results[col] = np.nan
        else:
            # find f_stat and p_value for the column
            f_stat, p_value = f_oneway(*groups)
            if p_value < 0.05: 
                anova_results[col] = [f_stat, p_value, 'sign.']
            else: 
                anova_results[col] = [f_stat, p_value, 'unsign.']
           

        
    # anova_results = {'gender' : [f_stat,p_value], 'education' : [f_stat, p_value] '}
    anova_df = pd.DataFrame(anova_results, index = ['f_stat', 'p_value', 'significance'])
    anova_df = anova_df.T.sort_values(by='f_stat', ascending=False)
    anova_df_sorted = anova_df.sort_values(by='f_stat', ascending=True)   
    if visual:
        # horizontal bar chart will show the importance and significance
        plt.figure(figsize=(width, height))
        bars = plt.barh(y=anova_df_sorted.index, width=anova_df_sorted['f_stat'], 
                 color=color) # 1 bar for each category

        # each bar has x,y coordinates, height, width
        if annot: 
            max_bar_width = anova_df_sorted['f_stat'].max()
            for i, bar in enumerate(bars): 
                # i = 0, bar = bar_1_gender
                width = bar.get_width() # width of the bar
                feature = anova_df_sorted.index[i] # category for the corresponding bar
                f = anova_df_sorted.loc[feature, 'f_stat'] # f stat value of corresponding category
                p = anova_df_sorted.loc[feature, 'p_value']
                sign = anova_df_sorted.loc[feature, 'significance']
                
                if width > 0.25*max_bar_width:
                    plt.text(width*0.5, bar.get_y() + bar.get_height()*0.5,
                            f'f_stat: {round(f,4)} \np_value: {round(p,4)} \n {sign}',
                            ha='center', va='center', color='white', fontsize=10)
                else: 
                    plt.text(width + 0.01*anova_df_sorted['f_stat'].max(), bar.get_y() + bar.get_height()*0.5,
                        f'f_stat: {round(f,4)} \np_value: {round(p,4)} \n {sign}',
                         ha='left', va='center', color='black', fontsize=10)


            
        plt.show()
                

    return anova_df 



            


    







































