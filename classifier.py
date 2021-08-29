import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier

# Do not change the code above
# These libraries may be all those you need
LABELS = ['setosa', 'versicolor', 'virginica']
# Insert your code below

class SoftmaxModel:
    """ A softmax classifier to fit, predict and score model 
    How to build the model (SoftmaxModel): It is just a normal class that will take an array 
    as coefficients during initialisation. During class initialisation, we will pass a 3 5 matrix 
    which will be used as the initial coefficient ( ) of the model.
    """ 
    def __init__(self, value):
        self.value = value #input_3_5_np_array
        self.coeff = None
        self.X_train = None
        self.y_train = None     
        self.epochs = None
        self.learning_rate = None
        self.prob_predict = None
        self.y_prob_max = None
    
    def fit(self, X_train, y_train, epochs, learning_rate):
        """ Fit model using softmax calculation (function) to find predict probability 
        and gradient descent to update the coefficient. Repeat softmax function and gradient descent to 
        update the predict probability and coefficient for each epoch time.
        How to train the model (fit): We will pass X_train, y_train, the number of epochs 
        (the number of times we will update the coefficient), and the learning rate to the fit function.
        The fit function will return the coefficient after training.
        """         
        self.coeff = self.value

        # Input - add bias '1' first column to X_train
        X_train = X_add_bias_col(X_train)

        for i in range(epochs):           
        
            # Softmax function
            prob_predict = softmax(X_train, self.coeff)
            #print('prob_predict:',prob_predict.shape, prob_predict)
            
            # Gradient Descent function
            self.coeff = gradient_descent(X_train, y_train, self.coeff, prob_predict, learning_rate)
            
        return self.coeff # Theata
        
    def predict(self, X_test):
        """ Predict the output labels. 
        How to predict: Prediction will take X_test as the data feature that needs to be predicted. 
        We will output a list of prediction labels based on the coefficient (an attribute of the model) 
        we trained during the last stage.
        """                
        # Input - add bias '1' first column to X_train
        X_test = X_add_bias_col(X_test)
        
        prob_test = softmax(X_test, self.coeff)
        #print('predict:', prob_test.shape, prob_test) # 47, 3
        
        # Get max vertical axis = 1
        self.y_prob_max = np.argmax(prob_test, axis=1)
        #####y_prob_max_onehot = onehot_encoding(self.y_prob_max, 3)
        #print('self.y_prob_max:', self.y_prob_max.shape, self.y_prob_max)                 
        #print('y_prob_max_onehot:', y_prob_max_onehot.shape, y_prob_max_onehot)

        # Decode LABELS y_prob_max_label is for display only
        y_prob_max_label = decode_onehot(self.y_prob_max)
        
        ###y_prob_max_label = np.array(LABELS)[np.array(pd.DataFrame(y_prob_max_label).idxmax(axis=1))]
        #print('y_prob_max_label:', y_prob_max_label)
        
        return y_prob_max_label
        #return self.y_prob_max
    
    def score(self, X_test, y_test):
        """ Accuracy score of the model 
        How to get the accuracy (score): We will compare the predicted labels with the ground true labels (y_test),
        to estimate the accuracy of the model prediction.
        """  
        
        # Mandatory: Call predict function # Do Not remove #
        y_prob_max_label = self.predict(X_test)
        
        #print('self.y_prob_max:', self.y_prob_max.shape, self.y_prob_max)
        y_test_max = np.argmax(y_test, axis=1)
        #print('y_test_max:',y_test_max.shape, y_test_max)
        
        accuracy = np.mean(np.equal(y_test_max, self.y_prob_max))        
        #print('accuracy:', accuracy)
        
        return accuracy
    
def pre_process(trainfile, testfile):
    """ To pre-process the input file and provide X features and y labels with one-hot encoding as output. """

    X_train, y_train = pre_process_file(trainfile)
    X_test, y_test = pre_process_file(testfile)
    
    return X_train, y_train, X_test, y_test

def pre_process_file(file):
    """ To pre-process the input file by removing unit cm from sepal length and sepal width and 
    converting mm to cm for sepal width and provide X features and y labels with one-hot encoding as output."""

    num_class = len(LABELS) # number of classes
    
    dataframe = pd.read_csv(file) 
    dataframe = dataframe.values
    
    # X array for features
    dataframe2 = dataframe[:,1:5]
    
    # Y array for labels
    dataframe3 = dataframe[:,5:6]

    # X features - Get and Round up 4 columns to 1 decimal point    
    for item in dataframe2:
        item[0] = round(float(item[0].replace('cm','').strip()),1)
        item[1] = round(float(item[1].replace('mm','').strip())/1000,1)
        item[2] = round(item[2],1)
        item[3] = round(item[3],1)
    
    X_array = np.array(dataframe2, dtype=np.float)
    
    # Y labels - One hot encoding
    for item in dataframe3:
        if item[0] == 'setosa':
            item[0] = int(0)
        elif item[0] == 'versicolor':
            item[0] = int(1)
        elif item[0] == 'virginica':
            item[0] = int(2)
    
    y_array = onehot_encoding(dataframe3, num_class)
    
    return X_array, y_array

def onehot_encoding(y, num_class):
    """ One-hot encoding for labels setosa as 0 0 1, versicolor as 0 1 0 and virginica as 1 0 0 """    
    y = np.asarray(y, dtype='int32')

    if len(y) > 1:
        y = y.reshape(-1)
        
    if not num_class:
        num_class = np.max(y) + 1

    y_matrix = np.zeros((len(y), num_class), dtype=int)
    y_matrix[np.arange(len(y)), y] = 1
    return y_matrix

def decode_onehot(y_prob_max):
    """ TO decode one-hot encoding for labels 0 as setosa, 1 as versicolor and 2 as virginica """        
    # Y labels - One hot decoding
    y_prob_max_label = y_prob_max.astype(str)

    # Modifying array values
    for x in np.nditer(y_prob_max_label, op_flags = ['readwrite']):
        if x == '0':
            x[...] = 'setosa'
        elif x == '1':
            x[...] = 'versicolor'
        elif x == '2':
            x[...] = 'virginica'

    y_prob_max_label = np.array(y_prob_max_label)

    return list(y_prob_max_label) # Return output as a list
        
def X_add_bias_col(X_train):
    """ Input - add bias '1' first column to X_train """
    # Input - add bias '1' first column to X_train
    Xrows, Xcols = np.shape(X_train) #103, 4
    X_train = np.hstack((np.ones((Xrows,1)),X_train)) # 103, 5
    #print('X_train:',X_train.shape, X_train)
    return X_train

def softmax(X_train, coeff):
    """ Softmax calculation to calculate Probability that y belongs to class k (has label k) — for the 3 species.
    Numerator: is the coefficients for belonging to class k, and X(i) is the feature values.
    Denominator: is the coefficients for belonging to class j, and X(i) is the feature values. 
    We sum up over all classes (3 in this case).
    For example: if is a 3 x 5 matrix filled with 1’s, then will be [1, 1, 1, 1, 1] (the first row), 
    and X(i) will be the values of the feature data (sepal_length, sepal_width, petal_length and petal_width) 
    plus one column to represent the bias, so it will read as [1, sepal_length, sepal_width, petal_length, petal_width]T. 
    We will then multiply them together.
    In the denominator we just need to perform exact same computation and add it all up PER EACH CLASS (K = 3 classes in this case).
    """
    # Theata, weight, prob of 3 classes
    #coeff = np.dot(coeff, X_train.T) # 3, 5 dot 5, 103 (after transpose) => 3, 103
    coeff = coeff @ X_train.T # 3, 5 dot 5, 103 (after transpose) => 3, 103
    coeff = coeff.T #=> 103, 3
    #print('softmax coeff:', coeff.shape, coeff) # CORRECT!

    numerator = np.exp(coeff) # 3,103 => z 103, 3
    #print('numerator:',numerator.shape, numerator)

    denominator = np.sum(np.exp(coeff),axis=1) # e_x.sum(axis=0) 
    #print('denominator:', denominator.shape, denominator)

    prob_predict = (numerator.T/denominator).T # 103, 3 divide by 1 => 103, 3
    #print('prob_predict:',prob_predict.shape, prob_predict)
    
    ###total_prob = np.sum(prob_predict, axis=1) # Verification: sum axis=1 is horizontal sum
    ###print('total_prob:', total_prob.shape, total_prob)
    
    return prob_predict

def gradient_descent(X_train, y_train, coeff_gd, prob_predict, learning_rate):
    """ Gradient descent to update the coefficient 
    The Predict Probability we get from the softmax function will be used to perform the gradient descent.
    X(i) is defined identically as for the previous state, and Plabel(y(i) = k) will be either 1 
    (if the i-th datum belongs to class k) or 0, usign the one hot encoding of pre-process.
    The learning rate is a scalar value.
    """
    # Gradient Descent function
    ############################
    # z = z + c1 * c2 * X(i)     
    # z is coefficient
    # c1 is y_labels with onehot encoding - prob predict (from softmax function)
    # c2 is learning rate
    # X(i) is X_features input
    ############################    
    #coeff_gd = coeff_gd + np.dot((y_train - prob_predict), X_train_5dim) *  learning_rate
    #print('y_train:', y_train.shape, y_train) # 103,3
    #print('prob_predict:', prob_predict.shape, prob_predict) # 103,3
        
    ### coeff_gd = np.add(coeff_gd, (np.dot(np.subtract(y_train, prob_predict).T, X_train_5dim) *  learning_rate))
    coeff_gd = np.add(coeff_gd, np.dot((y_train-prob_predict).T, X_train) *  learning_rate)
    #print('y_train - prob_predict:', a.shape, a) # 103,3 => transpose 3, 103 dot X 103, 5 => 3, 5
    #print('coeff_gd:', coeff_gd.shape, coeff_gd)
    
    return coeff_gd

def get_y_labels(y_labels):
    """ Get labels to be used in skl function """
    #LABELS = ['setosa', 'versicolor', 'virginica']
    max_label_index_df = pd.DataFrame(y_labels).idxmax(axis=1)
    max_label_index_matrix = np.array(max_label_index_df)
    y_decode_labels = np.array(LABELS)[max_label_index_matrix]

    return y_decode_labels
               
def skl(X_train, y_train, X_test, y_test, random_state, max_iter):
    """ sklearn model testing - the function will return the score of the sklearn model
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
    Input four numpy arrays, namely, X_train, y_train, X_test and y_test, as well as random_state and max_iter.
    How to construct the model: We just need to pass the values of random_state and max_iter 
    when we define a SGDClassifier. In this project, we use “log” as the loss function. 
    Tolerance will set to 1e-3 by default. 
    Fit and score will be easily retrieved by studying the material linked to above.
    """
    # To decode one-hot encoding y_train and y_test to return labels as output

    y_train_labels = get_y_labels(y_train)
    y_test_labels = get_y_labels(y_test)

    #print('y_train_labels:',y_train_labels)
    #print('y_test_labels:',y_test_labels)

    clf = SGDClassifier(max_iter=max_iter, tol=1e-3, loss='log', random_state=random_state)

    #To fit and score the model
    clf.fit(X_train, y_train_labels)
    return clf.score(X_test, y_test_labels)
        
        
def main():
    try:

        # Sample test cases from PDF       
        X_train, y_train, X_test, y_test = pre_process('train1.csv','test1.csv')
        print(X_train[:5])
        print(y_train[:5])
        
        X_train, y_train, X_test, y_test = pre_process('train2.csv', 'test2.csv')
        print(y_train[:5])
        print(X_train[:5])
        
        X_train, y_train, X_test, y_test = pre_process('train2.csv', 'test2.csv')
        # Initial coefficient is np.full((3,5), 0.5)
        model = SoftmaxModel(np.full((3, 5), 0.5))
        print(np.round(model.fit(X_train, y_train, 300, 1e-4), 3))
        
        X_train, y_train, X_test, y_test = pre_process('train3.csv', 'test3.csv')
        model = SoftmaxModel(np.full((3, 5), 0.5))
        print(np.round(model.fit(X_train, y_train, 500, 1.5e-4), 3))
        
        X_train, y_train, X_test, y_test = pre_process('train1.csv', 'test1.csv')
        model = SoftmaxModel(np.full((3, 5), 0.5))
        model.fit(X_train, y_train, 1000, 1e-4)
        print(model.predict(X_test))
        print(round(model.score(X_test, y_test),3))
        
        X_train, y_train, X_test, y_test = pre_process('train5.csv', 'test5.csv')
        model = SoftmaxModel(np.full((3, 5), 0.5))
        model.fit(X_train, y_train, 500, 1.5e-4)
        print(model.predict(X_test))
        print(round(model.score(X_test, y_test),3))
        
        X_train, y_train, X_test, y_test = pre_process('train4.csv', 'test4.csv');
        print(round(skl(X_train, y_train, X_test, y_test, 1, 1000), 3))
        
        X_train, y_train, X_test, y_test = pre_process('train3.csv', 'test3.csv');
        print(round(skl(X_train, y_train, X_test, y_test, 2, 500), 3))
        
        # Testing
        X_train, y_train, X_test, y_test = pre_process('train1.csv','test1.csv')        
        model = SoftmaxModel(np.full((3,5), 0.5))
        model.fit(X_train, y_train, 1000, 1e-4)
        print(np.round(model.fit(X_train, y_train,1000,1e-4),3))
        #print(np.round(model.fit(X_train, y_train,300,1e-4),3))
        #print('Predict:')
        model.predict(X_test)
        model.score(X_test, y_test)
        
        #Sklearn model testing
        ###print('Sklearn model testing')
        X_train, y_train, X_test, y_test = pre_process('train4.csv','test4.csv')
        #print('X_train4:',X_train, 'y_train4:', y_train)
        #print('X_test4:',X_test, 'y_test4:', y_test)

        print((skl(X_train, y_train, X_test, y_test, 1, 1000), 3))
        #print((skl(X_train, y_train, X_test, y_test, 2, 500), 3))

    ################### Exceptions ###################
    except ValueError:
        print('Caught ValueError')         
    except ZeroDivisionError:
        print('Caught ZeroDivisionError') 
    except TypeError:
        print('Caught TypeError')     
    except Exception:
        print('Caught Exception')
    except StopIteration:
        print('Caught StopIteration')
    except SystemExit:
        print('Caught SystemExit')
    except StandardError:
        print('Caught StandardError')
    except ArithmeticError:
        print('Caught ArithmeticError')
    except OverflowError:
        print('Caught OverflowError')
    except FloatingPointError:
        print('Caught FloatingPointError')
    except AssertionError:
        print('Caught AssertionError')
    except AttributeError:
        print('Caught AttributeError')
    except EOFError:
        print('Caught EOFError')
    except ImportError:
        print('Caught ImportError')
    except KeyboardInterrupt:
        print('Caught KeyboardInterrupt')
    except LookupError:
        print('Caught LookupError')
    except IndexError:
        print('Caught IndexError')
    except KeyError:
        print('Caught KeyError')
    except NameError:
        print('Caught NameError')
    except UnboundLocalError:
        print('Caught UnboundLocalError')
    except EnvironmentError:
        print('Caught EnvironmentError')
    except IOError:
        print('Caught IOError')
    except SyntaxError:
        print('Caught SyntaxError')
    except IndentationError:
        print('Caught IndentationError')
    except SystemError:
        print('Caught SystemError')
    except SystemExit:
        print('Caught SystemExit')
    except RuntimeError:
        print('Caught RuntimeError')
    except NotImplementedError:
        print('Caught NotImplementedError')    

if __name__ == '__main__':
    main()
