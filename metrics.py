import matplotlib.pyplot as plt


class IBLTMetrics:
    def __init__(self):
        """
        The IBLT by nature does not judge correctness; the IBLTMetrics class does, by using saved solution values.
        As such, all final metrics calculations are done with IBLTMetrics protected functions.
        Metrics are shown with report_metrics().
        """
        # The probabilistic stress as predicted by SVMe
        self.svm_calls = []
        # The probabilistic stress as predicted by kNN
        self.knn_calls = []
        # The probabilistic stress as decided by the judge
        self.final_calls = []
        # Source of result as selected by judge
        self.final_verdict = []
        # The deterministic stress, ie solution key
        self.deterministic = []
        # current # right / current n
        self.accuracy_by_n = []

        self.judge_accuracy = []

    def report_metrics(self, show_final_metrics=True,
                       show_judge_graph=True,
                       show_binary_graph=True,
                       show_performance_by_stress_graph=True):

        svm_accuracy = self._calculate_accuracy(self.svm_calls, self.deterministic)
        knn_accuracy = self._calculate_accuracy(self.knn_calls, self.deterministic)
        judge_accuracy = self._calculate_accuracy(self.final_calls, self.deterministic)

        if show_final_metrics:
            print("Total iterations: ", len(self.svm_calls))  # Should be length of svm calls
            print("Times kNN was right and was selected: ", self._count_knn_right_selected())
            print("Times kNN was right but not selected: ", self._count_knn_right_not_selected())
            print("Times kNN was wrong: ", self._count_knn_wrong())
            print("\n\n--- Final Accuracy ---\n")
            print("SVMe: ", svm_accuracy[-1])
            print("kNN: ", knn_accuracy[-1])
            print("Judge: ", judge_accuracy[-1])
            print("---------------------")
        if show_judge_graph:
            self._make_accuracy_graph(svm_accuracy, knn_accuracy, judge_accuracy)
        if show_binary_graph:
            self._make_binary_graph()
        if show_performance_by_stress_graph:
            self._make_performance_by_stress_graph()

    def _count_knn_right_selected(self):
        count = 0
        for i in range(len(self.knn_calls)):
            if self.knn_calls[i] == self.deterministic[i] and self.final_verdict[i] == "kNN":
                count += 1
        return count

    def _count_knn_right_not_selected(self):
        count = 0
        for i in range(len(self.knn_calls)):
            if self.knn_calls[i] == self.deterministic[i] and self.final_verdict[i] != "kNN":
                count += 1
        return count

    def _count_knn_wrong(self):
        count = 0
        for i in range(len(self.knn_calls)):
            if self.knn_calls[i] != self.deterministic[i]:
                count += 1
        return count

    def _display_accuracy_by_n(self, x):
        # Calculate the number of elements to display
        num_elements = len(self.accuracy_by_n)
        num_display = (num_elements - 1) // 10 * 10  # Exclude the last element

        # Display every 10th element
        elements_to_display = " ".join(str(self.accuracy_by_n[i]) for i in range(x, num_display + 1, 10))

        # Display the last element
        last_element = str(self.accuracy_by_n[-1])

        # Print all elements in a row
        print(elements_to_display, last_element)

    def _calculate_accuracy(self, answer_list, solution_list):
        """
        Calculate accuracy for each iteration.

        Parameters:
            answer_list (list): List of predicted values.
            solution_list (list): List of true values.

        Returns:
            accuracy_list (list): List of accuracy values for each iteration.
        """
        accuracy_list = []
        correct_count = 0

        for i in range(len(answer_list)):
            if answer_list[i] == solution_list[i]:
                correct_count += 1
            accuracy = correct_count / (i + 1)  # Calculate accuracy for current iteration
            accuracy_list.append(accuracy)

        return accuracy_list

    def _make_accuracy_graph(self, svm, knn, judge):

        # Generate iterations with a fixed interval (e.g., 1 iteration per data point)
        iterations = [i + 1 for i in range(len(svm))]

        # Create the plot and add multiple lines
        plt.plot(iterations, judge, linestyle='-', label='Judge')
        plt.plot(iterations, svm, linestyle='--', label='SVMe')
        plt.plot(iterations, knn, linestyle=':', label='kNN')

        # Add labels and title
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison over Iterations')

        # Add a legend
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.show()

    def _make_binary_graph(self):
        pass

    def _make_performance_by_stress_graph(self):
        pass

#
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     accuracy1 = [0.7, 0.8, 0.75, 0.85, 0.9, 0.88, 0.82, 0.78, 0.92, 0.95]
#     accuracy2 = [0.65, 0.72, 0.68, 0.78, 0.83, 0.81, 0.75, 0.71, 0.87, 0.91]
#     accuracy3 = [0.45, 0.52, 0.58, 0.88, 0.23, 0.31, 0.45, 0.61, 0.67, 0.21]
#
#     # Generate iterations with a fixed interval (e.g., 1 iteration per data point)
#     iterations = list(range(1, len(accuracy1) + 1))
#
#     # Alternatively, you can use a linear progression for iterations
#     # iterations = [i + 1 for i in range(len(accuracy))]
#
#     # Print the generated iterations
#     print(iterations)
#     # Create the plot and add multiple lines
#     plt.plot(iterations, accuracy1, marker='o', linestyle='-', label='Model 1')
#     plt.plot(iterations, accuracy2, marker='x', linestyle='--', label='Model 2')
#     plt.plot(iterations, accuracy3, marker='D', linestyle=':', label='Model 3')
#
#     # Add labels and title
#     plt.xlabel('Iterations')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy Comparison over Iterations')
#
#     # Add a legend
#     plt.legend()
#
#     # Display the plot
#     plt.grid(True)
#     plt.show()
