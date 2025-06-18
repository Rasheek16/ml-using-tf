def plot_image(image_data, dim=(28, 28)):
    import matplotlib.pyplot as plt

    image = image_data.reshape(dim)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()


def customKFold(X_train, y_train, model):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    skfolds = StratifiedKFold(n_splits=3)

    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(model)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))


def precision_recall_threshold_plot(thresholds, precisions, recalls, threshold):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))  # extra code – it's not needed, just formatting
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

    # extra code – this section just beautifies and saves Figure 3–5
    idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
    plt.plot(thresholds[idx], precisions[idx], "bo")
    plt.plot(thresholds[idx], recalls[idx], "go")
    plt.axis([-50000, 50000, 0, 1])
    plt.grid()
    plt.xlabel("Threshold")
    plt.legend(loc="center right")
    # save_fig("precision_recall_vs_threshold_plot")

    plt.show()


def precision_recall_plot(precisions, recalls, thresholds, threshold):
    import matplotlib.patches as patches  # extra code – for the curved arrow
    import matplotlib.pyplot as plt

    idx = (thresholds >= threshold).argmax()  # first index ≥ threshold

    plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

    # extra code – just beautifies and saves Figure 3–6
    plt.plot([recalls[idx], recalls[idx]], [0.0, precisions[idx]], "k:")
    plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
    plt.plot([recalls[idx]], [precisions[idx]], "ko", label="Point at threshold 3,000")
    plt.gca().add_patch(
        patches.FancyArrowPatch(
            (0.79, 0.60),
            (0.61, 0.78),
            connectionstyle="arc3,rad=.2",
            arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
            color="#444444",
        )
    )
    plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")
    # save_fig("precision_vs_recall_plot")

    plt.show()


def plot_roc_curve(threshold_for_90_precision, y_train, y_scores):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
    tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

    plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
    plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k:", label="Random classifier's ROC curve")
    plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")

    # extra code – just beautifies and saves Figure 3–7
    plt.gca().add_patch(
        patches.FancyArrowPatch(
            (0.20, 0.89),
            (0.07, 0.70),
            connectionstyle="arc3,rad=.4",
            arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
            color="#444444",
        )
    )
    plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
    plt.xlabel("False Positive Rate (Fall-Out)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.grid()
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower right", fontsize=13)
    # save_fig("roc_curve_plot")

    plt.show()


def plot_compare_pr_curves(
    recalls_clf_one,
    precisions_clf_one,
    recalls_clf_two,
    precisions_clf_two,
    lable_one,
    label_two,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

    plt.plot(recalls_clf_one, precisions_clf_one, "b-", linewidth=2, label=lable_one)
    plt.plot(recalls_clf_two, precisions_clf_two, "--", linewidth=2, label=label_two)

    # extra code – just beautifies and saves Figure 3–8
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")

    plt.show()
