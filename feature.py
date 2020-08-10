import numpy as np
import csv
import math
import sys


def read_dictionary(input):
    with open(input, "r") as file:
        reader = csv.reader(file, delimiter=' ')
        return {word:(int)(index) for word,index in reader}

def metric_1_read_data(input, dictionary, output_file):
    with open(input, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            output = {}
            write_data = line[0]
            for word in line[1].split(' '):
                value = dictionary.get(word, -1)
                value2 = output.get(value, -1)
                if value is not -1 and value2 is -1:
                    write_data += '\t' + (str)(value)+':'+(str)(1)
                    output[value] = 1
            output_data(write_data, output_file)

def metric_2_read_data(input, dictionary, output_file):
    with open(input, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            output = {}
            write_data = line[0]
            for word in line[1].split(' '):
                value = dictionary.get(word, -1)
                if value is not -1:
                    value2 = output.get(value, -1)
                    if value2 is -1:
                        output[value] = 1
                    else:
                        output[value] += 1
            for index in output.keys():
                if output[index] < 4:
                    write_data += '\t' + (str)(index)+':'+(str)(1)
            output_data(write_data, output_file)

def output_data(data, output_file):
    with open(output_file, "a") as output:
        output.write(data+'\n')

if __name__ == "__main__":
    input_train = sys.argv[1]
    input_valid = sys.argv[2]
    input_test = sys.argv[3]
    input_dictionary = sys.argv[4]
    output_train = sys.argv[5]
    output_valid = sys.argv[6]
    output_test = sys.argv[7]
    metric = sys.argv[8]

    dictionary = read_dictionary(input_dictionary)

    if (int)(metric) is 1:
        metric_1_read_data(input_train, dictionary, output_train)
        metric_1_read_data(input_valid, dictionary, output_valid)
        metric_1_read_data(input_test, dictionary, output_test)
    else:
        metric_2_read_data(input_train, dictionary, output_train)
        metric_2_read_data(input_valid, dictionary, output_valid)
        metric_2_read_data(input_test, dictionary, output_test)
