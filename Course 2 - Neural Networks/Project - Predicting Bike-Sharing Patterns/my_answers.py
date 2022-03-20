import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Original implementation had a lot of unnecessary transposes.
        # These were cleaned up in final versions, using print statements
        # to verify parity. Only the first iteration of each step is printed.
        debug_print = False
        self.debug_train = debug_print
        self.debug_forward = debug_print
        self.debug_backprop = debug_print
        self.debug_update_weights = debug_print
        self.debug_run = debug_print
                
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            if self.debug_train:
                print(f"TRAIN: X {X.shape}, y {y}")
                print()
                self.debug_train = False
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        # The hidden layer will use the sigmoid function for activations.
        hidden_inputs =  X.dot(self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        if self.debug_forward:
            print("FORWARD PASS")       
            print(f"hidden_inputs {hidden_inputs.shape} = X {X.shape} dot w_i2h {self.weights_input_to_hidden.shape}")
            print(f"sigmoid(hidden_inputs) = {hidden_outputs.shape}")            
            print(X.dot(self.weights_input_to_hidden))
        
        # TODO: Output layer - Replace these values with your calculations.
        # The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. 
        # That is, the activation function is f(x)=x
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        if self.debug_forward:
            print(f"final_inputs {final_inputs.shape} = hidden_outputs {hidden_outputs.shape} dot w_h2o {self.weights_hidden_to_output.shape}")
            print(f"final_outputs = final_inputs {final_outputs.shape}")
            print()
            self.debug_forward = False
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        # Sigmoid', i.e. f'(x) for f(x)=x is 1
        output_error_term = error
        
        if self.debug_backprop:
            print("BACKPROP")
            print(f"X: {X.shape}")
            print(f"error {error.shape} = y {y} - final_outputs {final_outputs.shape}")
            print(f"output_error_term = error {error.shape}")
        
        # TODO: Calculate the hidden layer's contribution to the error
        #hidden_error_orig = output_error_term.dot(self.weights_hidden_to_output.T)
        hidden_error = self.weights_hidden_to_output.dot(output_error_term)

        if self.debug_backprop:
            #print(f"hidden_error orig  {hidden_error_orig.shape} = output_error_term {output_error_term.shape} dot weights_hidden_to_output.T {self.weights_hidden_to_output.T.shape}")
            print(f"hidden_error final {hidden_error.shape} = weights_hidden_to_output {self.weights_hidden_to_output.shape} dot output_error_term {output_error_term.shape}")
            #print(f"hidden_error orig  {hidden_error_orig}")
            print(f"hidden_error final {hidden_error}")
        
        #hidden_error_term_orig = hidden_error_orig * hidden_outputs * (1-hidden_outputs)
        
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)

        if self.debug_backprop:
            #print(f"hidden_error_term orig  {hidden_error_term_orig.shape} = hidden_error {hidden_error_orig.shape} * hidden_outputs {hidden_outputs.shape} * (1-hidden_outputs {(1-hidden_outputs).shape})")
            print(f"hidden_error_term final {hidden_error_term.shape} = hidden_error {hidden_error.shape} * hidden_outputs {hidden_outputs.shape} * (1-hidden_outputs {(1-hidden_outputs).shape})")
            #print(f"hidden_error_term orig  {hidden_error_term_orig}")
            print(f"hidden_error_term final {hidden_error_term}")
            
            print(f"delta_weights_i_h orig {(hidden_error_term * X[:,None]).shape} += hidden_error_term {hidden_error_term.shape} * X[:,None] {X[:,None].shape}")            
            
            #print(f"delta_weights_h_o orig {(hidden_outputs[:,None].dot(output_error_term)).shape} += hidden_outputs[:,None] {hidden_outputs[:,None].shape} dot output_error_term {output_error_term.shape}")
            print(f"delta_weights_h_o final {(output_error_term * hidden_outputs[:,None]).shape} += output_error_term {output_error_term.shape} * hidden_outputs[:,None] {hidden_outputs[:,None].shape}")
            
            #print(f"delta_weights_h_o orig {(hidden_outputs[:,None].dot(output_error_term))[:, None]}")
            print(f"delta_weights_h_o final {output_error_term * hidden_outputs[:,None]}")
            print()
            self.debug_backprop = False
            
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:,None]
        
        # Weight step (hidden to output)
        #delta_weights_h_o += hidden_outputs[:,None].dot(output_error_term)[:, None] 
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        if self.debug_update_weights:
            print("UPDATE WEIGHTS")
            print(f"self.lr {self.lr} * delta_weights_h_o {delta_weights_h_o.shape} / n_records {n_records}")
            print(f"self.lr {self.lr} * delta_weights_i_h {delta_weights_i_h.shape} / n_records {n_records}")
            print()
            self.debug_update_weights = False
        
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        
        if self.debug_run:
            print("RUN")
            print(f"features shape = {features.shape}")
            
        # Oh boy, this had me stuck for longer than I'd like to admit.
        # This works: np.dot(pandas_dataframe, numpy_array)
        # This fails: pandas_dataframe.dot(numpy_array)
        # hidden_inputs = features.dot(self.weights_input_to_hidden) # signals into hidden layer
        
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        
        if self.debug_run:
            print(f"hidden_inputs is type: {type(hidden_inputs)}")
            print()
            self.debug_run = False
        
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # signals into final output layer
        # The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node.
        final_outputs = final_inputs # signals from final output layer 
             
        return final_outputs



#########################################################
# Set your hyperparameters here
##########################################################

# Final hyperparameters, from Test 6:
iterations = 4000
learning_rate = 0.5
hidden_nodes = 12
output_nodes = 1

# Test 7: Will an increase in iterations squeeze blood from a stone? Itr 4000 => 6000
# RESULTS converge to: Training loss: 0.058 ... Validation loss: 0.156
# Note: No
# iterations = 6000
# learning_rate = 0.5
# hidden_nodes = 12
# output_nodes = 1

# Test 6: increase hidden_nodes 10 => 12
# RESULTS converge to: Training loss: 0.063 ... Validation loss: 0.150
# iterations = 4000
# learning_rate = 0.5
# hidden_nodes = 12
# output_nodes = 1

# Test 5: reduce hidden_nodes 15 => 10
# RESULTS converge to: Training loss: 0.067 ... Validation loss: 0.153
# Note: No significant difference from 15 to 10 hidden nodes
# iterations = 4000
# learning_rate = 0.5
# hidden_nodes = 10
# output_nodes = 1

# Test 4: Reuse 0.5 learning rate and 15 hidden nodes, increase iterations 2000 => 4000
# RESULTS converge to: Training loss: 0.067 ... Validation loss: 0.151
# Note: validation loss was still decreasing when 2000 iterations completed
# iterations = 4000
# learning_rate = 0.5
# hidden_nodes = 15
# output_nodes = 1

# Test 3: Reuse 0.5 learning rate, reduce hidden nodes 25 => 15
# RESULTS converge to: Training loss: 0.067 ... Validation loss: 0.151
# Note: validation loss was still decreasing after 2000 iterations
# iterations = 2000
# learning_rate = 0.5
# hidden_nodes = 15
# output_nodes = 1

# Test 2, increase learning rate 0.5 => 0.8
# RESULTS converge to: Training loss: 0.534 ... Validation loss: 0.622
# Note: validation loss increased with 0.8 learning rate
# iterations = 2000
# learning_rate = 0.8
# hidden_nodes = 25
# output_nodes = 1

# Test 1, increase learning rate 0.2 => 0.5
# RESULTS converge to: Training loss: 0.143 ... Validation loss: 0.279
# iterations = 2000
# learning_rate = 0.5
# hidden_nodes = 25
# output_nodes = 1

# Test 0, pick some random values
# RESULTS converge to: Training loss: 0.271 ... Validation loss: 0.456
# iterations = 2000
# learning_rate = 0.2
# hidden_nodes = 25
# output_nodes = 1

