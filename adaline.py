import json
import random
from config import learning_rate, threshold, stop_condition

train_dataset = []
test_dataset = []

weights = []
bias = [0]

def initialize():
    global weights
    global bias
    global train_dataset
    global test_dataset

    weights = []
    for i in range(5):
        weights.append([])
        for j in range(5):
            weights[i].append(random.uniform(0.01, 0.05))     
    bias[0] = random.uniform(0.01, 0.05)

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

def train_test_split(dataset):
    global train_dataset
    global test_dataset
    
    selected = 0
    for i in range(len(dataset)):
        if i % 5 == 0:
            selected = random.randint(i, i+4)
            test_dataset.append(dataset[selected])
            if selected != i:
                train_dataset.append(dataset[i])
                
        elif i != selected:
            train_dataset.append(dataset[i])

def train():

    file = None
    
    try:

        initialize()

        file = open('dataset.txt', 'r')
        data_set = file.readline()
        data_set = json.loads(data_set)
        file.close()

        train_test_split(data_set)

        #run adaline algorithm:
        cur_change = 1
        lts_change = 2
        max_change = 0
        first_pass = True
        epochs = 0

        while abs(cur_change-lts_change) > stop_condition or first_pass:
            lts_change = cur_change
            max_change = 0
            first_pass = False

            for data in train_dataset:
                Yin = 0

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

            cur_change = max_change         
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
        
        return decode_label(activation(Yin))


    except:
        print("An Unexpected error occured!")
    finally:
        if file:
            file.close()
        

def accuracy_check():

    global test_dataset

    correct = 0

    for data in test_dataset:
        pred = test(data['features'])
        if pred.lower() == data['label']:
            correct += 1
    print(f"adaline accuracy is: {correct/len(test_dataset)*100}%")

    return correct/len(test_dataset)*100


def average_accuracy_check(repeat:int):
    avg = 0
    for i in range(repeat):
        train()
        avg += accuracy_check()

    print(f"avarage accuracy of adaline is {avg/repeat} %")

train()
average_accuracy_check(20)