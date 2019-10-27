
import os
from termcolor import colored
from helper import prepare_dataset, split_dataset, evaluate_features



def main():
    data, labels=prepare_dataset()
    train_data, test_data, train_label, test_label=hlp.split_dataset(data, labels, 0.2)
    train_label=train_label.astype('int')
    test_label=test_label.astype('int')
    svm_classifier(train_data, test_data, train_label, test_label, 'test')
    gnb(train_data, test_data, train_label, test_label, 'test')
    evaluate_features(train_data, train_label)



if __name__ == "__main__":
    # os.system("CLS")
    print(colored("\t\t\t\t\tStart of Executation", 'green'))
    main()
    print(colored("\t\t\t\t\tEnd of Executation", 'red'))
