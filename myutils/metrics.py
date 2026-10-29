from sklearn.metrics import *
from sklearn.metrics import confusion_matrix



def calculate_metrics(
        y_true,
        y_pred,
        y_prob):


    tn,fp,fn,tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()


    specificity=tn/(tn+fp+1e-10)



    return {


"Accuracy":
accuracy_score(
    y_true,
    y_pred
),


"Precision":

precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
),


"Recall":

recall_score(
    y_true,
    y_pred,
    pos_label=1
),


"F1":

f1_score(
    y_true,
    y_pred,
    pos_label=1
),


"ROC-AUC":
roc_auc_score(
    y_true,
    y_prob
),


"MCC":
matthews_corrcoef(
    y_true,
    y_pred
),


"Balanced Accuracy":
balanced_accuracy_score(
    y_true,
    y_pred
),


"Specificity":
specificity

}
