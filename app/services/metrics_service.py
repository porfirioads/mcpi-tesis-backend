import pandas as pd
import seaborn as sns
from app.schemas.algorithm_schemas import AlgorithmMetrics
from app.services.dataset_service import DatasetService
from app.patterns.singleton import SingletonMeta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,\
    precision_score, recall_score, roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')

sns.set_style("darkgrid")

dataset_service = DatasetService()


class MetricsService(metaclass=SingletonMeta):
    def get_metrics(self, y_test, y_pred) -> AlgorithmMetrics:
        accuracy = accuracy_score(y_test, y_pred)

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=1,
            average='binary'
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=1,
            average='binary'
        )

        sensitivity = recall_score(
            y_test,
            y_pred,
            pos_label=1,
            average='binary'
        )

        specificity = recall_score(
            y_test,
            y_pred,
            pos_label=-1,
            average='binary'
        )

        return AlgorithmMetrics(
            accuracy=accuracy,
            f1=f1,
            precision=precision,
            sensitivity=sensitivity,
            specificity=specificity
        )

    def plot_confusion_matrix(self, y_test, y_pred, title: str, file_path: str):
        cm = confusion_matrix(y_test, y_pred=y_pred.round())
        plt.figure()
        ax = plt.axes()
        sns.heatmap(cm, annot=True)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.xaxis.set_ticklabels(['negative', 'positive'])
        ax.yaxis.set_ticklabels(['negative', 'positive'])
        plt.savefig(file_path)
        plt.close()

    def plot_aucroc(self, model, x_test, y_test, title: str, file_path: str):
        ns_probs = [0 for _ in range(len(y_test))]
        # y_pred = model.predict_proba(x_test)[:, 1]
        md_probs = model.predict_proba(x_test)

        # keep probabilities for the positive outcome only
        md_probs = md_probs[:, 1]

        # calculate scores
        # ns_auc = roc_auc_score(y_test, ns_probs)
        md_auc = roc_auc_score(y_test, md_probs)

        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        md_fpr, md_tpr, _ = roc_curve(y_test, md_probs)

        # plot the roc curve for the model
        plt.figure()

        plt.plot(
            ns_fpr,
            ns_tpr,
            linestyle='--',
            label='No Skill',
            color='green'
        )

        plt.plot(md_fpr, md_tpr, label='AUCROC: {0:.2f}'.format(md_auc))
        plt.title("AUROC: {0}".format(title))
        plt.xlabel("False positive Rate (1 - specfity)")
        plt.ylabel("True positive rate (sensitivity)")
        plt.legend()
        plt.savefig(file_path)
        plt.close()
