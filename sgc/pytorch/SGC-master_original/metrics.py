from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    plot_roc(labels, preds)
    return micro, macro

def plot_roc(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot(fpr, tpr, 'b')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


    #fpr, tpr, _ = metrics.roc_curve(y_test,  preds)
    #auc = metrics.roc_auc_score(y_test, preds)
    #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    #plt.legend(loc=4)
    plt.savefig('plots/roc.pdf')
