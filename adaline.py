import json
from config import learning_rate, threshold, stop_condition

weights = []
bias = [0]

def initialize():
    for i in range(5):
        weights.append([])
        for j in range(5):
            weights[i].append(0)     
    bias[0] = 0

def activation(Yin):
    if Yin > threshold:
        return 1
    elif -threshold <= Yin <= threshold:
        return 0
    else:
        return -1

def encode_label(label):
    # 1 for x and -1 for o
    label = label.lower()
    if label == 'x':
        return 1
    elif label == 'o':
        return -1
    else:
        raise ValueError('please select a valid label among x or o') 

def decode_label(activation):
    if activation == 1:
        return 'X'
    elif activation == -1:
        return 'O'
    else:
        return 'NONE'

def train():

    file = None
    
    try:

        initialize()

        file = open('dataset.txt', 'r')
        data_set = file.readline()
        file.close()

        data_set = json.loads(data_set)

        #run adaline algorithm:
        max_change = 0
        first_pass = True
        epochs = 0

        while max_change > stop_condition or first_pass:
            first_pass = False
            for data in data_set:
                Yin = 0
                max_change = 0

                for i in range(5):
                    for j in range(5):
                        Yin += weights[i][j]*data['features'][i][j]
                Yin += bias[0]
                #update weights & bias:
                for i in range(5):
                    for j in range(5):
                        variation = learning_rate * data['features'][i][j] * (encode_label(data['label'])-Yin)
                        weights[i][j] += variation
                        if variation > max_change:
                            max_change = variation
                variation = learning_rate * (encode_label(data['label'])-Yin)
                bias[0] += variation
                if variation > max_change:
                    max_change = variation
                    
            print(f'max change is {max_change} in this epoch.')
            epochs += 1

        print(f'training successful through {epochs} epochs!')

    except ValueError as err:
        if err:
            print(err)
        else:
            print('An unExpected error occured!')

    finally:
        if file:
            file.close()


def test(test_data):
    file = None
    try:

        Yin = 0

        for i in range(5):
            for j in range(5):
                Yin += test_data[i][j]*weights[i][j]
        Yin += bias[0]

        print('Yin is:', Yin)

        return decode_label(activation(Yin))


    except:
        print("An Unexpected error occured!")
    finally:
        if file:
            file.close()
        