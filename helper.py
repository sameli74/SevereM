import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from termcolor import colored
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
plt.style.use('ggplot')
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support


c=np.arange(1, 101, 10)
gamma=np.array([1/108, 1/54, 1/27, 2/27, 4/27])
args=['LOCCOORD', 'ROAD_CLASS', 'TRAFFCTL', 'VISIBILITY',
        'LIGHT', 'RDSFCOND', 'Hour', 'LATITUDE', 'LONGITUDE', 'DATE','INJURY']


def check_distribution(data):
    unique, counts = np.unique(data, return_counts=True)
    for i in range(len(unique)):
        print("Number of [{0}]s: {1}".format(unique[i], counts[i]))
    return




def train_svm(train_data, test_data, train_label, test_label, c_val, type, g):
    if type=='gamma':
        classifier = svm.SVC(C=c_val, kernel='rbf', gamma=g)
    else:
        classifier = svm.SVC(C=c_val, kernel='rbf', gamma='auto')
    classifier.fit(train_data, train_label)
    predicted_label = classifier.predict(test_data)
    acc=accuracy_score(predicted_label, test_label)
    return acc

def plot_acc(avg_err, type):
    global c
    global gamma
    if type=='c':
        x=c
    elif type=='gamma':
        x=gamma
    fig=plt.figure(1)
    ax = fig.gca()
    ax.set_xlim([min(x), max(x)])
    if type=='c':
        ax.set_xlabel('C')
        ax.set_title('C vs Accuracy')
    elif type=='gamma':
        ax.set_xticks(x)
        ax.set_xlabel('Gamma')
        ax.set_title('Gamma vs Accuracy')
    plt.grid()
    ax.set_ylabel('Accuracy')
    plt.plot(x, avg_err, 'r')
    plt.show()

def kfold(data, label, c_val, type, gamma):
    j=0
    err=np.zeros((1, 10))
    kfold = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in kfold.split(data):
        test_label=label[test_index]
        test_data=data[test_index, :]
        train_data=data[train_index, :]
        train_label=label[train_index]
        err[0, j]=train_svm(train_data, test_data, train_label, test_label, c_val, type, gamma)
        j=j+1
    return err

def evaluate(iter, data, label , c, type, err, *args, **kwargs):
    gamma = kwargs.get('gamma', None)
    if iter==0:
        err=kfold(data, label, c, type, gamma)
    else:
        err=np.append(err, kfold(data, label, c, type, gamma), axis=0)
    return err


def evaluate_c(data, label, type):
    err=np.array([])
    global c
    i=0
    for c_val in c:
        print("C: {0}".format(c_val))
        err=evaluate(i, data, label, c_val, type, err)
        i=i+1
    return err

def evaluate_gamma(data, label, type, c):
    global gamma
    err=np.array([])
    i=0
    for g in gamma:
        print("gamma: {0}, C: {1}".format(g, c))
        err=evaluate(i, data, label, c, type, err, gamma=g)
        i=i+1
    return err

    return err
def train_model(data, label, type, *args, **kwargs):
    err=np.array([])
    c = kwargs.get('best_c', None)
    i=0
    if type=='c':
        err=evaluate_c(data, label, type)
    elif type=='gamma':
        err=evaluate_gamma(data, label, type, c)
    elif type=='all':
        err= evaluate(i, data, label, c, type, err)
    avg_err=np.mean(err, axis=1)
    if type=='all':
        print(avg_err)
    else:
        plot_acc(avg_err, type)


def prepare_dataset():
    dataset=read_data()
    dataset=remove_useless_features(dataset)
    dataset=prepare_injury(dataset)
    dataset=prepare_loccoord(dataset)
    dataset=prepare_road_class(dataset)
    dataset=prepare_traffic_ctl(dataset)
    dataset=prepare_visibility(dataset)
    dataset=prepare_light(dataset)
    dataset=prepare_road_cond(dataset)
    print_useful_features(dataset)

    labels=dataset['INJURY'].values
    dataset=dataset.drop('INJURY', 1)
    hour=dataset['Hour'].values
    dataset=dataset.drop('Hour', 1)
    lat=dataset['LATITUDE'].values
    dataset=dataset.drop('LATITUDE', 1)
    long=dataset['LONGITUDE'].values
    dataset=dataset.drop('LONGITUDE', 1)
    dataset.DATE = pd.to_datetime(dataset.DATE)
    dataset.DATE = dataset.DATE.dt.date
    dataset['DATE'] = pd.to_datetime(dataset['DATE'])
    dataset['DATE'] = dataset['DATE'].map(lambda x:x.month)
    month=dataset['DATE'].values
    dataset=dataset.drop('DATE', 1)
    data=dataset.values
    enc=OneHotEncoder()
    data=np.array(enc.fit_transform(data).toarray())
    new_data=np.zeros((data.shape[0], data.shape[1]+4))
    new_data[:, :-4]=data[:, :]
    new_data[:, -4]=month
    new_data[:, -3]=hour
    new_data[:, -2]=lat
    new_data[:, -1]=long
    data=new_data
    return data, labels

def split_dataset(dataset, labels, ratio):
    dataset, labels=shuffle(dataset, labels, random_state=10)
    x_train, x_test, y_train, y_test=train_test_split(dataset, labels,test_size=ratio, random_state=42)
    return x_train, x_test, y_train, y_test


def show_content(arg, color, dataset):
    print('-'*50)
    print(arg+':')
    classes = pd.value_counts(dataset[arg])
    print (colored(classes, color))
    return

def print_useful_features(dataset):
    selected=['LOCCOORD', 'ROAD_CLASS', 'TRAFFCTL', 'VISIBILITY',
            'LIGHT', 'RDSFCOND','INJURY']
    for arg in selected:
        show_content(arg, 'yellow', dataset)
    return

def plot_injury(classes):
    size=17
    ax=classes.plot(kind = 'bar', rot=0 ,color="coral", fontsize=size, width=.2)
    labels = ['None', 'Major', 'Minor', 'Minimal', 'Fatal']
    plt.title("Distribution of Injuries", fontsize=size)
    plt.xticks(range(5), labels)
    plt.xlabel("Injury Level", fontsize=size)
    plt.ylabel("Frequency", fontsize=size)
    for i in ax.patches:
        ax.text(i.get_x()-0.1, i.get_height()+20, str(i.get_height()), color='dimgrey', fontsize=size)
    plt.show()
    return

def modify_injury(dataset, blank):
    dataset['INJURY'].replace(to_replace="None", value=-1, inplace=True)
    dataset['INJURY'].replace(to_replace="Minor", value=-1, inplace=True)
    dataset['INJURY'].replace(to_replace="Minimal", value=-1, inplace=True)
    dataset['INJURY'].replace(to_replace="Major", value=+1, inplace=True)
    dataset['INJURY'].replace(to_replace="Fatal", value=+1, inplace=True)
    lst=dataset.index[dataset['INJURY'] == blank].tolist()
    dataset=dataset.drop(dataset.index[lst])
    dataset=dataset.reset_index(drop=True)
    return dataset

def read_data():
    dataset= pd.read_csv('./Dataset/KSI.csv')
    return dataset

def remove_useless_features(dataset):
    global args
    features=list(dataset.columns.values)
    for column in features:
        if not (column in args):
            dataset = dataset.drop(column, 1)
    return dataset


def prepare_injury(dataset):
    classes = pd.value_counts(dataset['INJURY'])
    blank=classes.index[2]
    classes=classes.drop(blank)
    dataset=modify_injury(dataset, blank)
    return dataset

def remove_row(field, column_name, dataset):
    lst=dataset.index[dataset[column_name] == field].tolist()
    dataset=dataset.drop(dataset.index[lst])
    dataset=dataset.reset_index(drop=True)
    return dataset

def prepare_loccoord(dataset):
    classes = pd.value_counts(dataset['LOCCOORD'])
    blank=classes.index[2]
    dataset['LOCCOORD'].replace(to_replace="Park, Private Property, Public Lane", value=blank, inplace=True)
    dataset['LOCCOORD'].replace(to_replace="Entrance Ramp Westbound", value=blank, inplace=True)
    dataset=remove_row(blank, 'LOCCOORD', dataset)
    return dataset

def prepare_road_class(dataset):
    dataset['ROAD_CLASS'].replace(to_replace="Expressway Ramp", value="", inplace=True)
    dataset['ROAD_CLASS'].replace(to_replace="Major Arterial Ramp", value="", inplace=True)
    dataset['ROAD_CLASS'].replace(to_replace="Laneway", value="", inplace=True)
    dataset=remove_row("", 'ROAD_CLASS', dataset)
    return dataset

def prepare_traffic_ctl(dataset):
    classes = pd.value_counts(dataset['TRAFFCTL'])
    blank=classes.index[6]
    dataset['TRAFFCTL'].replace(to_replace="Yield Sign", value=blank, inplace=True)
    dataset['TRAFFCTL'].replace(to_replace="Traffic Gate", value=blank, inplace=True)
    dataset['TRAFFCTL'].replace(to_replace="Police Control", value=blank, inplace=True)
    dataset['TRAFFCTL'].replace(to_replace="School Guard", value=blank, inplace=True)
    dataset['TRAFFCTL'].replace(to_replace="Streetcar (Stop for)", value=blank, inplace=True)
    dataset=remove_row(blank, 'TRAFFCTL', dataset)
    # print(dataset.shape)
    return dataset
def prepare_visibility(dataset):
    classes = pd.value_counts(dataset['VISIBILITY'])
    blank=classes.index[8]
    dataset['VISIBILITY'].replace(to_replace="Strong wind", value=blank, inplace=True)
    dataset['VISIBILITY'].replace(to_replace="Drifting Snow", value=blank, inplace=True)
    dataset['VISIBILITY'].replace(to_replace="Freezing Rain", value=blank, inplace=True)
    dataset['VISIBILITY'].replace(to_replace="Fog, Mist, Smoke, Dust", value=blank, inplace=True)
    dataset['VISIBILITY'].replace(to_replace="Other", value=blank, inplace=True)
    dataset=remove_row(blank, 'VISIBILITY', dataset)
    return dataset

def prepare_light(dataset):
    classes = pd.value_counts(dataset['LIGHT'])
    blank=classes.index[8]
    dataset=remove_row(blank, 'LIGHT', dataset)
    return dataset

def prepare_road_cond(dataset):
    dataset['RDSFCOND'].replace(to_replace="Spilled liquid", value="", inplace=True)
    dataset['RDSFCOND'].replace(to_replace="Loose Sand or Gravel", value="", inplace=True)
    dataset['RDSFCOND'].replace(to_replace="Packed Snow", value="", inplace=True)
    dataset['RDSFCOND'].replace(to_replace="Ice, Smoke, Dust", value="", inplace=True)
    dataset['RDSFCOND'].replace(to_replace="Ice", value="", inplace=True)
    dataset['RDSFCOND'].replace(to_replace="Other", value="", inplace=True)
    dataset=remove_row("", 'RDSFCOND', dataset)
    classes = pd.value_counts(dataset['RDSFCOND'])
    blank=classes.index[4]
    dataset=remove_row(blank, 'RDSFCOND', dataset)
    return dataset

def evaluate_classifier(train_data, test_data, train_label, test_label, classifier, acc_type):
    if acc_type=='train':
        train_acc=evaluate_classifier_phase(train_data, test_data, train_label, test_label, classifier, acc_type)
        print("training accuracy:{0}".format(train_acc))
    elif acc_type=='test':
        test_acc=evaluate_classifier_phase(train_data, test_data, train_label, test_label, classifier, 'test')
        print("testing accuracy:{0}".format(test_acc))
    return

def evaluate_classifier_phase(train_data, test_data, train_label, test_label, classifier, phase):
    classifier.fit(train_data, train_label)
    if phase=='train':
        predicted_label = classifier.predict(train_data)
        acc=accuracy_score(predicted_label, train_label)
        # cr=confusion_matrix(train_label, predicted_label)
    elif phase=='test':
        predicted_label = classifier.predict(test_data)
        acc=accuracy_score(predicted_label, test_label)
        target_names = ['-1', '1']
        print(classification_report(test_label, predicted_label, target_names=target_names))
    return acc

def svm_classifier(train_data, test_data, train_label, test_label, acc_type):
    classifier = svm.SVC(kernel='rbf')
    evaluate_classifier(train_data, test_data, train_label, test_label, classifier, acc_type)
    return

def DT_classifier(train_data, test_data, train_label, test_label, acc_type):
    classifier=DecisionTreeClassifier()
    evaluate_classifier(train_data, test_data, train_label, test_label, classifier, acc_type)
    return

def random_forest(train_data, test_data, train_label, test_label, acc_type):
    classifier=RandomForestClassifier()
    evaluate_classifier(train_data, test_data, train_label, test_label, classifier, acc_type)
    return
def gnb(train_data, test_data, train_label, test_label, acc_type):
    classifier=GaussianNB(var_smoothing=1000)
    evaluate_classifier(train_data, test_data, train_label, test_label, classifier, acc_type)
    return
def knn(train_data, test_data, train_label, test_label, acc_type):
    classifier=KNeighborsClassifier(n_neighbors=1)
    evaluate_classifier(train_data, test_data, train_label, test_label, classifier, acc_type)
    return

def evaluate_features(X,y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    print(indices)
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    lst=[]
    stds=[]
    lst.append(sum(importances[0:2]))
    stds.append(np.sqrt(np.sum(np.square(std[0:2]))/2))
    lst.append(sum(importances[2:7]))
    stds.append(np.sqrt(np.sum(np.square(std[2:7]))/5))
    lst.append(sum(importances[7:12]))
    stds.append(np.sqrt(np.sum(np.square(std[7:12]))/5))
    lst.append(sum(importances[12:15]))
    stds.append(np.sqrt(np.sum(np.square(std[12:15]))/3))
    lst.append(sum(importances[15:23]))
    stds.append(np.sqrt(np.sum(np.square(std[15:23]))/8))
    lst.append(sum(importances[23:27]))
    stds.append(np.sqrt(np.sum(np.square(std[23:27]))/4))
    lst=np.array(lst+importances[27:32].tolist())
    stds=np.array(stds+std[27:32].tolist())
    indices2 = np.argsort(lst)[::-1]
    args=np.array(['Location\nType', 'Road\nClass', 'Traffic\nControl', 'Visibility',
            'Light', 'Road\nCondition', 'Hour', 'Latitude', 'Longitude', 'Month','Injury'])

    size=27
    plt.figure(figsize=(20,8))
    plt.rcParams.update({'font.size': size})
    plt.title("Importance of Features (%)",fontsize=size)
    plt.bar(range(lst.shape[0]), lst[indices2],
           color="r", yerr=stds[indices2], align="center")
    plt.xticks(range(lst.shape[0]), args[indices2], rotation=-60, color='black')
    plt.yticks(color='black')
    plt.tight_layout()
    plt.savefig('demo.eps', transparent=True)
    plt.show()
    return
