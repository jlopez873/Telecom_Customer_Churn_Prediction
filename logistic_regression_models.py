## Import libraries/packages
## Import libraries/packages
import numpy as np
import pandas as pd
from scipy.stats import zscore, pointbiserialr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

## Read in data
print('Reading Data From "churn_clean.csv"', end='\n\n')
df = pd.read_csv('churn_clean.csv').reset_index(drop=True)

## View data types
print('Reviewing Data', end='\n\n')
df.info()

## Rename survey columns
df.rename({
    'Item1':'TimelyResponse',
    'Item2':'TimelyFixes',
    'Item3':'TimelyReplacements',
    'Item4':'Reliability',
    'Item5':'Options',
    'Item6':'RespectfulResponse',
    'Item7':'CourteousExchange',
    'Item8':'ActiveListening'
}, axis=1, inplace=True)

## View summary statistics
df.describe()

## Drop less meaningful columns
df = df.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State', 'County', 'Zip', 'Lat', 
              'Lng', 'Area', 'TimeZone', 'Job', 'Marital', 'Gender', 'Email', 'Multiple', 
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'PaperlessBilling',
              'PaymentMethod'], axis=1)

## Create copy of dataframe
df1 = df.copy()

## Check for missing values
print('\nMissing Values Found:', sum(df1.isna().sum()), end='\n\n')

## Check for duplicate values
print('Duplicate Values Found:', len(df1) - df1.duplicated().value_counts()[0], end='\n\n')

## Check for outliers
df1.describe()

## Separate object variables
df2 = pd.DataFrame([df1[col] for col in df1.columns if df1[col].dtype != 'object']).transpose()

## Normalize data and exclude outliers
df2 = df2[zscore(df2).abs() < 3]

## Count outliers
print('Outliers Found:', sum(df2.isna().sum()), '\nRemoving Outliers', end='\n')

## Drop outlier values
df2.dropna(inplace=True)
print('Outliers Remaining:', sum(df2.isna().sum()), end='\n\n')

## Measure data loss
lost = ((len(df1) - len(df2))/len(df1))*100
remaining = 100 - lost
print('Lost Data: {}%\nRemaining Data: {}%'.format(round(lost, 2), remaining), end='\n\n')

## Combine dataframes
df = df.loc[df2.index]
df1 = df1.loc[df2.index]

## Reset index values
df = df.reset_index(drop=True)
df1 = df1.reset_index(drop=True)

## Calculate summary statistics for dependent variable
df1.Churn.describe()

## Calculate summary statistics for independent variables
independent_vars = pd.DataFrame(columns=['min', 'max', 'std', 'mean', 'median', 'mode'])
for col in df1.columns:
    if df1[col].dtype != object:
        independent_vars.loc[col] = [
            min(df1[col]),
            max(df1[col]),
            np.std(df1[col]),
            df1[col].mean(),
            df1[col].median(),
            df1[col].mode().values[0]
        ]

## Identify variables with Yes or No values
nominal = ['Churn', 'Techie', 'Port_modem', 'Tablet', 'Phone', 'StreamingTV', 'StreamingMovies']

## Store value distribution in data frame
nominal_df = pd.DataFrame([df[var].value_counts() for var in nominal])

## Get contract and internet service distributions
contract = pd.DataFrame(df.Contract.value_counts())
internet_service = pd.DataFrame(df.InternetService.value_counts())

## Create x-axis labels
x_contract = np.arange(len(contract.index))
x_internet_service = np.arange(len(internet_service.index))

## Define function to get grouped count for ordinal values
def get_ordinal_vals(var, data):
    ## Create dataframe of grouped value counts
    df = pd.DataFrame(df1.groupby([var, 'Churn'])[var].sum())
    ## Create no and yes lists of values
    no = []
    yes = []
    ## Iterate through each rating and store value counts
    for i in range(1,7):
        no.append(df.transpose()[i]['No'].values[0])
        yes.append(df.transpose()[i]['Yes'].values[0])
    ## Create dataframe to store simplified values
    temp_df = pd.DataFrame(index=np.arange(1,7), columns=['No', 'Yes'])
    temp_df['No'] = no
    temp_df['Yes'] = yes
    ## Return simplified dataframe
    return temp_df

## Get counts for churn by ordinal rating
timely_response = get_ordinal_vals('TimelyResponse', df1)
timely_fixes = get_ordinal_vals('TimelyFixes', df1)
timely_replacements = get_ordinal_vals('TimelyReplacements', df1)
reliability = get_ordinal_vals('Reliability', df1)
options = get_ordinal_vals('Options', df1)
respectful_response = get_ordinal_vals('RespectfulResponse', df1)
courteous_exchange = get_ordinal_vals('CourteousExchange', df1)
active_listening = get_ordinal_vals('ActiveListening', df1)

## Identify continuous vraiables
cont = ['Population', 'Children', 'Age', 'Income', 'Outage_sec_perweek', 'Contacts', 
        'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'TimelyResponse', 
        'TimelyFixes', 'TimelyReplacements', 'Reliability', 'Options', 'RespectfulResponse', 
        'CourteousExchange', 'ActiveListening']

## Calculate correlation coefficients
corr_coef = pd.DataFrame(index=cont, columns=['coef', 'pvalue'])
for var in cont:
    cont_var = df1[var]
    cat_var = df1['Churn']
    # Convert categorical variable to binary (0 or 1)
    binary_var = pd.get_dummies(cat_var, drop_first=True)
    # Calculate point-biserial correlation coefficient
    corr, pvalue = pointbiserialr(cont_var, binary_var)
    corr_coef.loc[var] = corr[0], pvalue

## Create InternetDSL and InternetFiberOptic columns
dsl = []
fiber = []
for i in df1.InternetService:
    if i == 'DSL':
        dsl.append(1)
        fiber.append(0)
    elif i == 'Fiber Optic':
        dsl.append(0)
        fiber.append(1)
    else:
        dsl.append(0)
        fiber.append(0)

## Assign values ro InternetDSL and InternetFiberOptic columns
df1['InternetDSL'] = dsl
df1['InternetFiberOptic'] = fiber

## Encode InternetService column
internet_service = {'DSL':'Yes', 'Fiber Optic':'Yes', 'None':'No'}
df1.InternetService.replace(internet_service, inplace=True)

## Initiate label encoder
le = LabelEncoder()

## Encode variables
for col in df1.columns:
    if 'Yes' in df1[col].values:
        df1[col] = le.fit_transform(df1[col])

## Encode contract variable
contract = {'Month-to-month':0, 'One year':1, 'Two Year':2}
df1.Contract.replace(contract, inplace=True)

## Review changes
df1.info()

## Store clean data as CSV
print('\nSaving data to "churn_logistic_regression.csv"', end='\n\n')
df1.to_csv('churn_logistic_regression.csv')

## Define target and explanatory variables
X, y = df1.drop('Churn', axis=1), df1['Churn']

## Split training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y)

## Define initial features
current_features = X.columns.values

## Define function to initiate and test models
def init_mod(X_train, X_test, y_train, y_test, features):
    ## Initiate model
    mod = LogisticRegression(fit_intercept=True).fit(X_train[features], y_train)
    ## Review coefficients
    coef_df = pd.DataFrame(mod.coef_[0], index=features, columns=['coef'])
    print(coef_df)
    ## Test model
    y_pred = mod.predict(X_test[features])
    return mod, y_pred

## Initiate and test model
mod1, pred1 = init_mod(X_train, X_test, y_train, y_test, current_features)

## Define function to score models
def score_mod(y_test, y_pred, features):
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), features

## Create dataframe for model metrics
results = pd.DataFrame(columns=['f1', 'accuracy', 'features'])

## Log model metrics
results.loc['InitialModel'] = score_mod(y_test, pred1, current_features)

## Select feature with the highest correlation to the target variable
feature_1 = corr_coef.index[corr_coef.coef.abs() == max(corr_coef.coef.abs())][0]
                                 
## Create list of features already included
selected_features = [feature_1]
                                 
## Create dataframe to store model metrics
step_results = pd.DataFrame(columns=['f1', 'accuracy', 'features'])

print('Performing Step-Forward Feature Selection', end='\n\n')
## Create list of features per iteration
for i in range(len(corr_coef)-1):
    ## Initiate and test model
    mod, pred = init_mod(X_train, X_test, y_train, y_test, selected_features)
    ## Log results
    step_results.loc[i] = score_mod(y_test, pred, selected_features[:i+1])
    ## Drop already selected features
    dropped = corr_coef.drop(selected_features)
    ## Select next feature
    selected_features.append(dropped.index[dropped.coef.abs() == max(dropped.coef.abs())][0])

## Review model results
step_results = step_results.sort_values('f1', ascending=False).reset_index(drop=True)
print('\nStep-Forward Feature Selection Results:')
print(step_results, end='\n\n')

## Select features from best performing model
current_features = step_results.features[0]

## Define function to print remaining features
def print_features(features):
    for feature in features:
        print(feature)
    print('Features Remaining:', len(features), end='\n\n')

## View remaining features
print('Reduced features:')
print_features(current_features)

## Recreate model
mod2, pred2 = init_mod(X_train, X_test, y_train, y_test, current_features)

## Log model results
results.loc['Model2'] = score_mod(y_test, pred2, current_features)

def get_vif(X):
    ## Add constant
    X['const'] = 1.0
    ## Create dataframe to store vif values
    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='const']
    return vif
                                 
## Calculate vif to check for multicollinearity
print('Calculating Variance Inflation Factors', end='\n\n')
vif = get_vif(X[current_features])
print('Variance Inflation Factors:')
print(vif, end='\n\n')

## Remove variable with highest VIF
if len(vif['Variable'][vif['VIF'] > 5]) > 0:
    print('Removing Variable With Highest VIF', end='\n\n')
current_features = vif[vif['VIF'] != vif['VIF'].max()].Variable.values

## Re-calculate vif
vif = get_vif(X[current_features])
print('Variance Inflation Factors:')
print(vif, end='\n\n')

## Initiate and test reduced model
mod3, pred3 = init_mod(X_train, X_test, y_train, y_test, current_features)

## Log model metrics
results.loc['Model3'] = score_mod(y_test, pred3, current_features)

## Print results
print('Initial F1 Score:', results.f1.InitialModel)
print('Initial Accuracy Score', results.accuracy.InitialModel)
print('Initial Features:', len(results.features.InitialModel))
print('Final F1 Score:', results.f1.Model3)
print('Final Accuracy Score', results.accuracy.Model3)
print('Final Features:', len(results.features.Model3))
print('\nFinal features used:')
for n in results.features.Model3:
    print(n)