import numpy as np
import csv
import math
import sys

def read_dictionary(input):
    with open(input, "r") as file:
        reader = csv.reader(file, delimiter=' ')
        return {word:(int)(index) for word,index in reader}

def initialize_theta(dictionary):
    theta = {}
    for key in dictionary.keys():
        theta[dictionary[key]] = 0
    theta[39176] = 0
    return theta

def read_data(input_file):
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        labels = []
        output = []
        for line in reader:
            data = []
            data.append((int)(line[0]))
            temp = {}
            for i in range(1, len(line)):
                temp[(int)(line[i].split(':')[0])] = (int)(line[i].split(':')[1])
            temp[39176] = 1
            data.append(temp)
            output.append(data)
    return output

def dot_product(a, b):
    output = 0.0
    for k in b.keys():
        output += a[k]*b[k]
    return output


def negative_loglikelihood(input, theta):
    j_output = 0
    for i in range(0, len(input)):
        dot = dot_product(theta, input[i][1])
        print(dot)
        sigmoid = 1/((float)(1+np.exp(-1*dot)))
        if input[i][0] is 0:
            j_output += np.log(1-sigmoid)
        else:
            j_output += np.log(sigmoid)
    j_output = (-1/(float)(len(input)))*j_output
    return j_output

def stocastic_gradientdescent(inputl, theta, epoch, validationinput):
    alpha = 0.1
    for num in range(epoch):
        for i in range(0, len(inputl)):
            dot = dot_product(theta, inputl[i][1])
            for k in inputl[i][1].keys():
                theta[k] += alpha*(inputl[i][0]-(math.exp(dot)/(1+math.exp(dot))))
        #output_loglikelihood(str(num) + '\t' + str(negative_loglikelihood(inputl, theta)), "negative_loglikelihood_train.txt")
        #output_loglikelihood(str(num) + '\t' + str(negative_loglikelihood(validationinput, theta)), "negative_loglikelihood_valid.txt")
    return theta

def predict(input, theta):
    labels = []
    error = 0
    for i in range(0, len(input)):
        dot = dot_product(theta, input[i][1])
        p = np.exp(dot)/(1+math.exp(dot))
        if p >= 0.5:
            predicted_class = 1
        else:
            predicted_class = 0
        if input[i][0] is not predicted_class:
            error += 1
        labels.append(predicted_class)
    error = error/((float)(len(input)))
    return labels, error

def output(data, output_file):
    with open(output_file, "a") as output:
        for row in data:
            output.write(str(row)+'\n')
    return

def output_loglikelihood(data, output_file):
    with open(output_file, "a") as output:
        output.write(data+'\n')
    return

if __name__ == "__main__":
    input_train = sys.argv[1]
    input_valid = sys.argv[2]
    input_test = sys.argv[3]
    input_dictionary = sys.argv[4]
    output_train = sys.argv[5]
    output_test = sys.argv[6]
    metric = sys.argv[7]
    epoch = (int)(sys.argv[8])

    train_data = read_data(input_train)
    valid_data = read_data(input_valid)
    test_data = read_data(input_test)

    theta = initialize_theta(read_dictionary(input_dictionary))

    theta = stocastic_gradientdescent(train_data, theta, epoch, valid_data)

    train_labels, train_error = predict(train_data, theta)

    test_labels, test_error = predict(test_data, theta)

    with open(metric, "a") as metrics_output:
        metrics_output.write("error(train): "+str(train_error)+'\n')
        metrics_output.write("error(test): "+str(test_error)+'\n')

    output(train_labels, output_train)
    output(test_labels, output_test)
