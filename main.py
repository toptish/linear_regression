"""
Linear regression program with implementation of gradient descent algorithm
"""
import sys
import csv
import os.path
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt


class ValidationError(RuntimeError):
    pass


class ValidateData:
    pass


class CsvGetData:
    """
    Open and reads input file, stores values to self.dataset
    and self.dataset_no_header
    """

    def __init__(self, filename):
        self.filename = filename
        self.dataset = []
        self.dataset_no_header = []

    def validate_file(self):
        """

        :return:
        """
        if not os.path.isfile(self.filename):
            raise RuntimeError("Csv file " + self.filename + "doesn't exist")
        if not os.access(self.filename, os.R_OK):
            raise RuntimeError("Access denied for " + self.filename)
        with open(self.filename, 'r') as csv_file:
            try:
                csv_data = csv.reader(csv_file, delimiter=",")
                for row in csv_data:
                    self.dataset.append(row)
            except ValueError:
                raise RuntimeError("File" + self.filename + "cannot be read")

    def remove_dataset_headers(self):
        """

        :return:
        """
        self.dataset_no_header = [self.dataset[i] for i in range(1, len(self.dataset))]

    def transform_to_float(self):
        """

        :return:
        """
        self.dataset_no_header = [list(map(float, data_pair)) for data_pair in self.dataset_no_header]


class ReadArguments:
    """
    Reads and parses command line program arguments
    """

    def __init__(self):
        self.file_name = ""
        self.plot = False
        self.custom_learn_rate = False
        self.learning_rate = 0.1

    def parse_arguments(self):
        parser = argparse.ArgumentParser('Linear regression input arguments')
        parser.add_argument('filename', type=str, help='Filename for csv dataset')
        parser.add_argument('-lr', type=float, default=0.1, help='Set custom learning rate (default: 0.1)')
        parser.add_argument('-plot', type=bool, default=False, help='Plot the dataset and linear regression result')
        arguments = parser.parse_args()
        if arguments.__contains__('filename'):
            print(arguments)
            self.file_name = arguments.filename
        self.learning_rate = arguments.lr
        self.plot = arguments.plot


@dataclass
class RegressionCoefficients:
    def __init__(self):
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.temp_theta0 = 1.0
        self.temp_theta1 = 1.0



class LinearRegression:
    """
    Linear regression class using gradient descent algorithm
    """

    def __init__(self, dataset: list, learning_rate):
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.coeff = RegressionCoefficients()
        self.mean_se = self.mean_square_error()
        self.mean_square_error_prev = 0.0
        self.mse_delta = self.mean_se

    def estimate_result(self, param):
        """

        :param param:
        :return:
        """
        return self.coeff.temp_theta0 + self.coeff.temp_theta1 * param

    def set_min_max(self):
        self.min_x = MathStat.min([x[0] for x in self.dataset])
        self.min_y = MathStat.min([x[1] for x in self.dataset])
        self.max_x = MathStat.max([x[0] for x in self.dataset])
        self.max_y = MathStat.max([x[1] for x in self.dataset])

    def standardize(self):
        self.set_min_max()
        for line in self.dataset:
            line[0] = (line[0] - self.min_x) / \
                (self.max_x - self.min_x)
            line[1] = (line[1] - self.min_y) / \
                (self.max_y - self.min_y)

    def mean_square_error(self):
        """

        :return:
        """
        estimated_values = [self.estimate_result(data_pair[0]) for data_pair in self.dataset]
        real_values = [data_pair[1] for data_pair in self.dataset]
        return MathStat.mean_square_error(estimated_values, real_values)

    def get_gradient_theta0(self):
        """

        :return:
        """
        error_sum = 0.0
        for data_pair in self.dataset:
            error_sum += (self.estimate_result(data_pair[0]) - data_pair[1])
        # print(error_sum/len(self.dataset))
        return self.learning_rate * (error_sum / len(self.dataset))

    def get_gradient_theta1(self):
        """

        :return:
        """
        error_sum = 0.0
        for data_pair in self.dataset:
            error_sum += (self.estimate_result(data_pair[0]) - data_pair[1]) * data_pair[0]
        return self.learning_rate * (error_sum / len(self.dataset))

    def train_model(self):
        self.standardize()
        while self.mse_delta > 0.0000001 or self.mse_delta < -0.0000001:
        # for _ in range(0, 10000):
            self.coeff.theta0 = self.coeff.temp_theta0
            self.coeff.theta1 = self.coeff.temp_theta1
            self.coeff.temp_theta0 -= self.get_gradient_theta0()
            self.coeff.temp_theta1 -= self.get_gradient_theta1()
            print(self.coeff.theta0, self.coeff.theta1)
            self.mean_square_error_prev = self.mean_se
            self.mean_se = self.mean_square_error()
            self.mse_delta = self.mean_se - self.mean_square_error_prev
        self.save_coefficients()

    def save_coefficients(self):
        f = open("indexes.csv", "w+")
        f.write("%f, %f" % (self.coeff.theta0, self.coeff.theta1))
        f.close()


class MathStat:
    @staticmethod
    def min(datalist):
        """

        :param datalist:
        :return:
        """
        min_value = sys.float_info.max
        for number in datalist:
            min_value = number if number < min_value else min_value
        return min_value

    @staticmethod
    def max(datalist):
        """

        :param datalist:
        :return:
        """
        max_value = sys.float_info.min
        for number in datalist:
            max_value = number if number > max_value else max_value
        return max_value

    @staticmethod
    def mean_square_error(estimated_values, real_values):
        """

        :param estimated_values:
        :param real_values:
        :return:
        """
        square_err_sum = 0
        for estimated_value, real_value in zip(estimated_values, real_values):
            error = estimated_value - real_value
            error_square = error ** 2
            square_err_sum += error_square
        return square_err_sum / len(estimated_values)


class PlotResult:
    """

    """

    def __init__(self, dataset, theta0, theta1):
        self.x_plt = [data[0] for data in dataset]
        self.y_plt = [data[1] for data in dataset]
        self.theta0 = theta0
        self.theta1 = theta1
        self.min_x = MathStat.min(self.x_plt)
        self.max_x = MathStat.max(self.x_plt)
        self.min_y = MathStat.min(self.y_plt)
        self.max_y = MathStat.min(self.y_plt)
        self.x_plt_standardized = []
        self.y_plt_standardized = []

    def standardize(self):
        """

        :return:
        """
        self.x_plt_standardized = [self.min_x + (self.max_x - self.min_x) * x for x in self.x_plt]
        self.y_plt_standardized = [self.min_y + (self.max_y - self.min_y) * y for y in self.y_plt]

    def run_plot(self):
        plt.ion()
        fig, ax = plt.subplots()
        plt.grid = True
        # plt.plot(self.x_plt, self.y_plt, 'ro')
        ax.scatter(self.x_plt, self.y_plt)
        self.plot_linear_regression()
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # plt.show()
        plt.ioff()
        plt.show()

    def plot_linear_regression(self):
        estimated_y = [self.theta0 + self.theta1 * x for x in self.x_plt]
        plt.plot(self.x_plt, estimated_y, "red")


def main():
    try:
        input_args = ReadArguments()
        input_args.parse_arguments()
        csv_data = CsvGetData(input_args.file_name)
        csv_data.validate_file()
        csv_data.remove_dataset_headers()
        csv_data.transform_to_float()
        regression = LinearRegression(csv_data.dataset_no_header, input_args.learning_rate)
        regression.train_model()
        plot_data = PlotResult(csv_data.dataset_no_header, regression.coeff.theta0,
                               regression.coeff.theta1)
        plot_data.run_plot()


    except RuntimeError as e:
        print(e)


if __name__ == '__main__':
    main()
