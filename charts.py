from io import BytesIO
from base64 import b64encode

import numpy as np
from sklearn import metrics
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

def show_charts(data):
	label_pos_num = int(data.get("label_pos_num", "100"))
	label_neg_num = int(data.get("label_neg_num", "900"))
	prob_pos_mean = float(data.get("prob_pos_mean", "0.6"))
	prob_pos_std = float(data.get("prob_pos_std", "0.1"))
	prob_neg_mean = float(data.get("prob_neg_mean", "0.4"))
	prob_neg_std = float(data.get("prob_neg_std", "0.1"))
	threshold = float(data.get("threshold", "0.5"))
	seed = int(data.get("seed", "0"))
	normalize = data.get("normalize")
	if normalize == 'none':
		normalize = None

	np.random.seed(seed)
	y_true = np.concatenate([
		np.repeat(1, label_pos_num),
		np.repeat(0, label_neg_num),
	])
	y_prob_pos = np.random.normal(loc=prob_pos_mean, scale=prob_pos_std, size=label_pos_num)
	y_prob_neg = np.random.normal(loc=prob_neg_mean, scale=prob_neg_std, size=label_neg_num)
	y_prob = np.concatenate([y_prob_pos, y_prob_neg])

	fig, axs = plt.subplots(2, 2, figsize=(9,9))
	show_hist(y_prob_pos, y_prob_neg, threshold=threshold, ax=axs[0, 0])
	show_confusion(y_true, y_prob, threshold=threshold, normalize=normalize, ax=axs[0, 1])
	show_roc(y_true, y_prob, ax=axs[1, 0])
	show_prc(y_true, y_prob, ax=axs[1, 1])

	img = BytesIO()
	fig.savefig(img, format='png', transparent=False, dpi=80, bbox_inches="tight")
	plot_url = b64encode(img.getvalue()).decode()
	return f'<img src="data:image/png;base64,{plot_url}">'

def make_axes(ax):
	if ax is None:
		# plt.figure()
		# ax = plt.axes()
		fig, ax = plt.subplots()
	return ax

def show_hist(y_prob_pos, y_prob_neg, threshold=0.5, bins=20, ax=None):
	ax = make_axes(ax)
	ax.hist(y_prob_neg, bins=bins, color='r', alpha=0.6, label='Negative')
	ax.hist(y_prob_pos, bins=bins, color='g', alpha=0.6, label='Positive')
	ax.axvline(threshold, color='b', lw=1, label='Threshold')
	ax.legend(loc="upper right")
	ax.set_title('Histogram of predicted probabilities')
	
def show_confusion(y_true, y_prob, threshold=0.5, normalize=None, ax=None):
	ax = make_axes(ax)
	y_pred = (y_prob >= threshold).astype(int)
	cm = metrics.confusion_matrix(y_true, y_pred, normalize=normalize)
	metrics.ConfusionMatrixDisplay(cm, display_labels=["False", "True"]).plot(ax=ax, values_format='.5g')
	ax.set_title('Confusion matrix')
	
def show_roc(y_true, y_prob, ax=None):
	ax = make_axes(ax)
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
	roc_auc = metrics.auc(fpr, tpr)
	metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='').plot(ax=ax)
	ax.set_title('Receiver operating characteristic')

def show_prc(y_true, y_prob, ax=None):
	ax = make_axes(ax)
	precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true, y_prob)
	average_precision = metrics.average_precision_score(y_true, y_prob)
	metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision, estimator_name='').plot(ax=ax)
	ax.set_title('Precision-recall')
