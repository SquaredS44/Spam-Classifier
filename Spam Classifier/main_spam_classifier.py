import scipy.io as sio
from emailFeatures import emailFeatures
from processEmail import processEmail
import scipy
from sklearn import svm
import numpy as np


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)


def get_vocab():  # vocabulary list (from txt file)
    vocab_list = []
    with open('vocab.txt', 'r') as f:
        for line in f:
            for word in line.split():
                if not has_numbers(word):
                    vocab_list.append(word)
    f.close()
    return vocab_list

print('\nPreprocessing sample email (emailSample1.txt)\n')

# extract features
file = open('emailSample1.txt','r')
#file = open('emailSample2.txt','r')
file_contents = file.read()
word_idx = processEmail(file_contents)

# print word_index
print('Word Indices: \n')
print(word_idx)
print('\n\n')


features = emailFeatures(word_idx)

# print features
print('Length of feature vector:', len(features))
print('Number of non-zero entries:', sum(features > 0))
print('Program paused. Press enter to continue.\n')

# train svm
# load train set
mat = scipy.io.loadmat('spamTrain.mat')
x = mat["X"]
y = mat["y"]
# load test set
mat = scipy.io.loadmat('spamTest.mat')
xtest = mat["Xtest"]
ytest = mat["ytest"]

#pos = np.array([x[i] for i in xrange(x.shape[0]) if y[i] == 1])
#neg = np.array([x[i] for i in xrange(x.shape[0]) if y[i] == 0])
#print('Total number of training emails = ',x.shape[0])
#print('Number of training spam emails = ',pos.shape[0])
#print('Number of training nonspam emails = ',neg.shape[0])


print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')


linear_svm = svm.SVC(C=0.1, kernel='linear')
linear_svm.fit(x, y.flatten())


train_predictions = linear_svm.predict(x).reshape((y.shape[0],1))
train_acc = 100. * float(sum(train_predictions == y))/y.shape[0]
print('Training accuracy = %0.2f%%' % train_acc)

test_predictions = linear_svm.predict(xtest).reshape((ytest.shape[0],1))
test_acc = 100. * float(sum(test_predictions == ytest))/ytest.shape[0]
print('Test set accuracy = %0.2f%%' % test_acc)

vocab_list = get_vocab()

weights_idx = linear_svm.coef_.argsort() #get index of sorted (low2high) weights
idx2 = weights_idx[0][::-1] # reverse indexes (high2low)
words = get_vocab() #get vocab list
for i in range(15):
    print(linear_svm.coef_[0,idx2[i]],words[idx2[i]]) #print highest 15 weights and the corresponding word

#topword_indices = np.argsort(linear_svm.coef_[0])[::-1][:15]
#for i in topword_indices:
    #print(vocab_list[i], linear_svm.coef_[0,i])
#print()

filename = 'notSpam.txt'

# MY EMAIL YOOO LOTS OF FUN
my_file = open(filename)
my_file_contents = my_file.read()
my_word_idx = processEmail(my_file_contents)
my_x = emailFeatures(my_word_idx)
p = linear_svm.predict(np.array(my_x).reshape(1, -1))
print("Spam Classification:", p)
print("(1 indicates spam, 0 indicates not spam)")