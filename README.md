# neural_network
## A multilayer neural network classifier to classify whether a given message is SPAM or HAM. It is implemented from scratch in python without using any neural net library.

1. The dataset has 5574 messages, each annotated as SPAM or HAM.
2. Pre-processing the data includes: (i) Breaking each message into tokens (any sequence of characters separated by blanks, tabs, returns, dots, commas, colons and dashes can be considered as tokens) (ii) Removing a standard set of English stopwords, (iii) Applying Porter stemming.
3. The dataset is divided into train(80%) and test(20%).
4. The architecture includes an input layer(max 2000 words), a hidden layer with 100 neurons, again a hidden layer with 50 neurons and then an output layer.
5. Two different activation function have been used tanh and sigmoid.
6. In the last part the output layer is replaced with a softmax layer, rest inner layers have sigmoid activation.
