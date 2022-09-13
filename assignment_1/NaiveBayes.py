import math

class NaiveBayes:
    def __init__(self, train, test, n_folds) -> None:
        self.train = train
        self.test = test
        self.n_folds = n_folds

    def calculate_probability(self, x, mean, stdev):
        if stdev == 0: return 0
        exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
 
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        l = len(dataset) // n_folds

        for i in range(n_folds):
            dataset_split.append(dataset[i*l:(i+1)*l])
        return dataset_split

    def summarize_dataset(self, dataframe, alpha = 1, beta = 3):
        grouped = dataframe.groupby('Segmentation')

        summaries = dict()
        for name, grp in grouped:
            summaries[name] = list()
            for col in grp.columns:
                summaries[name].append((grp[col].mean(), grp[col].std(), len(grp[col]), alpha*grp[col].mean()+beta*grp[col].std()))
        return summaries
    

    def calculate_class_probabilities(self, summaries, data):
        total_rows = 0
        for label in summaries:
            total_rows += summaries[label][0][2]
        probabilities = dict()

        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)

            for i in range(len(class_summaries)):
                mean, stdev, _, _ = class_summaries[i]
                pr = self.calculate_probability(data[i], mean, stdev)
                if(pr > 0): probabilities[class_value] *= float(pr)

        return probabilities

    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        result = correct / float(len(actual)) * 100.0
        return result

    def fit(self):
        folds = self.cross_validation_split(self.train, self.n_folds)
        global_scores = -1
        optimal_summary = None
        scores = list()
        for fold in folds:
            train_set = self.train
            train_set.drop(fold.index)
            test_set = fold

            summarize = self.summarize_dataset(train_set)
            predictions = []

            for i in range(len(test_set)):
                
                probabilities = self.calculate_class_probabilities(summarize, test_set[i: i+1].values.flatten().tolist())
                best_label, best_prob = None, -1
                
                for class_value, probability in probabilities.items():
                    if best_label is None or probability > best_prob:
                        best_prob = probability
                        best_label = class_value
                
                predictions.append(best_label)

            actual = fold['Segmentation'].values.tolist()
            accuracy = self.accuracy_metric(actual, predictions)

            if accuracy > global_scores:
                global_scores = accuracy
                optimal_summary = summarize

            scores.append(accuracy)
        return scores, optimal_summary

    def get_test_accuracy(self, test_set, summarize):
        predictions = []
        for i in range(len(test_set)):
            probabilities = self.calculate_class_probabilities(summarize, test_set[i: i+1].values.flatten().tolist())
            best_label, best_prob = None, -1
            for class_value, probability in probabilities.items():
                if best_label is None or probability > best_prob:
                    best_prob = probability
                    best_label = class_value

            predictions.append(best_label)
        actual = test_set['Segmentation'].values.tolist()
        accuracy = self.accuracy_metric(actual, predictions)
        return accuracy