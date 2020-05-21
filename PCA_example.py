import numpy
import glob
def main():
    datasets = glob.glob("MNIST-MIX-all/*.npz")
    for dataset in datasets:
        data = numpy.load(dataset)
        training_digits = data['X_train']
        training_labels = data['y_train']
        test_digits = data['X_test']
        test_labels = data['y_test']
        tol = 1
        training_digits = training_digits.reshape((training_digits.shape[0], training_digits.shape[1]*training_digits.shape[2]))
        mean_training_data = numpy.mean(training_digits, axis=0)
        training_digits = training_digits - numpy.outer(numpy.ones(training_digits.shape[0]), mean_training_data)
        test_digits -= numpy.outer(numpy.ones(test_digits.shape[0]), mean_training_data)
        covariance_matrix = numpy.cov(training_digits, rowvar=False)
        eigenvals, eigenvectors = numpy.linalg.eigh(covariance_matrix)
        eigenvectors = eigenvectors[:, numpy.abs(eigenvals[:]) > tol]
        training_data = numpy.matmul(training_digits, eigenvectors)

    pi = 1
if __name__ == "__main__":
    main()
