#importing python modules
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
from plotly import graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import IPython
from IPython.display import display, HTML, Image
import plotly.io as pio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
#==================================================================================================================

#===============================================importing the dataset==============================================
data = pd.read_excel('dataset.xlsx')
#================================================viewing all column================================================
#print(data.head(5))
#================================================Cleaning up unnecessary columns===================================
df = data.drop(['What do you seek in a game?', 'Timestamp','Can you please select the GPU that you are currently using?','What is your RAM size?','What is your preferred CPU model?','Preference'], axis=1)

df = df.rename(columns={"What is your age limit?": "Age Limit",
                        'What genres game do you like?' : 'Genre',
                   "How long do you play games in average?": 'Playtime (min)', 
                   'Please select your internet speed.': 'Internet Speed (Mbps)',
                   'Can you please select an estimate of your monthly gaming cost(Including purchasing Game and Game Items , In app Purchase, And Net Bill)?' : 'Monthly Cost (BDT)'})

df['Monthly Cost (BDT)'] = df['Monthly Cost (BDT)'].replace({'\(BDT\)':''}, regex=True)
df['Monthly Cost (BDT)'] = df['Monthly Cost (BDT)'].replace({'I play cracked Games \(Could be Fitgirl, Codex, RG-Mechanics, Black Box, Skidrow, Reloaded etc\)':'0'}, regex=True)
df['Monthly Cost (BDT)'] = df['Monthly Cost (BDT)'].replace({'k':'000'}, regex=True)
df['Monthly Cost (BDT)'] = df['Monthly Cost (BDT)'].replace({'500-2000':'1500'}, regex=True)
df['Monthly Cost (BDT)'] = df['Monthly Cost (BDT)'].replace({'3-6000':'5000'}, regex=True)
df['Monthly Cost (BDT)'] = df['Monthly Cost (BDT)'].replace({'15-26000':'22000'}, regex=True)
df['Monthly Cost (BDT)'] = df['Monthly Cost (BDT)'].replace({'10-14000':'10000'}, regex=True)
df['Monthly Cost (BDT)'] = df['Monthly Cost (BDT)'].replace({'7-9000':'7500'}, regex=True)

df['Internet Speed (Mbps)'] = df['Internet Speed (Mbps)'].replace({'mbps':''}, regex=True)
df['Internet Speed (Mbps)'] = df['Internet Speed (Mbps)'].replace({'Gigabit':'1000'}, regex=True)

df['Playtime (min)'] = df['Playtime (min)'].str.replace("1-1.5 hours","80")
df['Playtime (min)'] = df['Playtime (min)'].str.replace("2-3hours","150")
df['Playtime (min)'] = df['Playtime (min)'].str.replace("4-5hours","270")
df['Playtime (min)'] = df['Playtime (min)'].str.replace("30-50min","45")
df['Playtime (min)'] = df['Playtime (min)'].str.replace("6-8hours","420")
df['Playtime (min)'] = df['Playtime (min)'].str.replace(r"9.*s","550")


df['Age Limit'] = df['Age Limit'].str.replace("+","")
df['Age Limit'] = np.random.randint(8, 60, df.shape[0])




conditions = [
    (df['Playtime (min)'].astype(float) >= 80) & (df['Monthly Cost (BDT)'].astype(float) <= 1500),
    (df['Playtime (min)'].astype(float) >= 150) & (df['Monthly Cost (BDT)'].astype(float) <= 7500),
    (df['Playtime (min)'].astype(float) >= 420) & (df['Monthly Cost (BDT)'].astype(float) <= 22000)
    ]

# create a list of the values we want to assign for each condition
values = ['Not Addict', 'Slightly Addicted', 'Addicted']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Condition'] = np.select(conditions, values)


conditions2 = [
    (df['Playtime (min)'].astype(float) >= 80) & (df['Monthly Cost (BDT)'].astype(float) <= 1500),
    (df['Playtime (min)'].astype(float) >= 150) & (df['Monthly Cost (BDT)'].astype(float) <= 7500),
    (df['Playtime (min)'].astype(float) >= 420) & (df['Monthly Cost (BDT)'].astype(float) <= 22000)
    ]

# create a list of the values we want to assign for each condition
values2 = ['0', '0.5', '1']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Condition In Value'] = np.select(conditions2, values2)

df = df.dropna()
#print(df['Age Limit'])
#df.to_excel('test.xlsx')

def configure_plotly_browser_state():
    display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
            requirejs.config({
                paths: {
                base: '/static/base',
                plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
            });
            </script>
            '''))

#init_notebook_mode(connected=True)

configure_plotly_browser_state()

labels = ['Addict', 'Slightly Addicted', 'Not Addicted']

count_not_addict = len(df[df['Playtime (min)']=='80'])
count_not_addict = len(df[df['Playtime (min)']=='45'])


count_slightly_addicted = len(df[df['Playtime (min)']=='150'])
count_slightly_addicted = len(df[df['Playtime (min)']=='270'])

count_addict = len(df[df['Playtime (min)']=='420'])
count_addict = len(df[df['Playtime (min)']=='550'])

values = [count_not_addict, count_slightly_addicted, count_addict]
# values = [20,50]

trace = go.Pie(labels=labels,
               values=values,
               textfont=dict(size=19, color='#FFFFFF'),
               opacity = 0.7,
               marker=dict(
                   colors=['blue', 'green', 'red']
               )
              )

layout = go.Layout(title = '<b>Addiction Distribution Based On Playtime</b>')
chart = [trace]
fig = go.Figure(data=chart, layout=layout)

plt.figure(figsize=[15, 6])
#iplot(fig)


#print("Error")
#fig.write_image("plots/addiction_based_on_playtime.pdf")
#fig.write_image("addiction_based_on_playtime.png")

#print("Worked")

sns.set_theme(style="darkgrid")
plt.figure(figsize=[15, 6])
sns.countplot(x="Monthly Cost (BDT)", data=df, saturation = 1, linewidth = 2, edgecolor = (0, 0, 0))
plt.title('Montly Cost Of Gaming', fontsize = 24)
plt.xlabel('Monthly Cost(BDT)', fontsize=16)
plt.ylabel('Count', fontsize=16)

#plt.savefig("plots/cost.pdf")

sns.set_theme(style="whitegrid")
plt.figure(figsize=[20, 6])
sns.countplot(y="Playtime (min)", data=df, saturation = 0.8, linewidth = 2, hue = 'Monthly Cost (BDT)')

plt.title('Playtime with respect to Preferences', fontsize = 24)
plt.ylabel('Playtime (min)', fontsize=16)
plt.xlabel('Count', fontsize=16)

#plt.savefig("plots/playtime.pdf")


#========================================== SKLEARN PART ==========================================================
df_m = df.copy()

# Convert these variables into categorical variables
df_m["Age Limit"] = df_m["Age Limit"].astype('category').cat.codes
df_m["Playtime (min)"] = df_m["Playtime (min)"].astype('category').cat.codes
df_m["Monthly Cost (BDT)"] = df_m["Monthly Cost (BDT)"].astype('category').cat.codes
df_m["Genre"] = df_m["Genre"].astype('category').cat.codes
# Target Variable
df_m['Condition_label'] = pd.Categorical(df_m['Condition'])
df_m['Condition_label'] = df_m['Condition_label'].astype('category')
df_m["Condition_label"] = df_m["Condition_label"].astype('category').cat.codes
print("working 1")
#Legend
#legend = dict( enumerate(df['Condition_label'].cat.categories ) )
print("working 2")
# Label Encoding
col = ['Age Limit', 'Playtime (min)', 'Internet Speed (Mbps)','Genre',
       'Monthly Cost (BDT)', 'Condition']
le = LabelEncoder()
for name in col:
    le.fit(df_m['Age Limit'].astype(str))
    df_m['Age Limit'] = le.transform(df_m['Age Limit'].astype(str))

# Changing data types
df_m["Playtime (min)"] = pd.to_numeric(df_m["Playtime (min)"])
df_m["Internet Speed (Mbps)"] = pd.to_numeric(df_m["Internet Speed (Mbps)"])


#print(legend)
df_m.dropna(subset = col, inplace=True)



y = df_m['Condition_label']
X = df_m.drop('Condition', axis = 1)
X = X.drop('Condition_label', axis = 1)

# Splitting up the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)

# Check accuracy of Logistic Model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(penalty='l2', C=1, max_iter = 20000)

LR_score = LR.fit(X_train, y_train)
print ("Logistic Regression Accuracy is : %2.2f" % accuracy_score(y_test, LR.predict(X_test)))

print ("\n\n ---------------- Logistic Regression Model ----------------")
print ('############################################################')
print(classification_report(y_test, LR.predict(X_test)))

# Decision Tree Model
DT = tree.DecisionTreeClassifier(
    max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.02
    )
DT_score = DT.fit(X_train,y_train)
print ("Decision Tree Accuracy is : %2.2f" % accuracy_score(y_test, DT.predict(X_test)))

print ("\n\n ----------------- Decision Tree Model -----------------")
print ('############################################################')
print(classification_report(y_test, DT.predict(X_test)))

# Random Forest Model
RF = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=30, 
    class_weight="balanced",
    min_weight_fraction_leaf=0.02 
    )
RF_score = RF.fit(X_train, y_train)
print ("Decision Tree Accuracy is : %2.2f" % accuracy_score(y_test, RF.predict(X_test)))

print ("\n\n ----------------- Random Forest Model -----------------")
print ('############################################################')
print(classification_report(y_test, RF.predict(X_test)))

# Ada Boost
AB = AdaBoostClassifier(n_estimators=600, learning_rate=0.1)
AB_score = AB.fit(X_train,y_train)

print ("Ada Boost Model Accuracy is : %2.2f" % accuracy_score(y_test, AB.predict(X_test)))

print ("\n\n ----------------- Ada Boost Model -----------------")
print ('############################################################')
print(classification_report(y_test, AB.predict(X_test)))