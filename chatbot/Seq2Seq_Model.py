"""
Model to predict the next sentence given an input sequence
To create the model, you need only to call 
"self.model = Model(self.args, self.textData)"
as:
self.model: the model instance in the whole work
self.textdata: can be got through "self.textData = TextData(self.args)"
"""
import tensorflow as tf
from chatbot.textdata import Batch

"""
Single layer perceptron algorithm implementation
Project input tensor on the output dimension
"""
class ProjectionOp:

    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W = tf.get_variable('weights',shape,dtype=dtype)
            self.b = tf.get_variable('bias',shape[1], initializer=tf.constant_initializer(), dtype=dtype)

    #Function to get the weight and bias of the model.
    def getWeights(self):
        return self.W, self.b

    # the equation of the classifier itself.
    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


"""
Implementation of a seq2seq model.
Architecture:
Encoder/decoder
2 LTSM layers
"""
class Model:
   

    def __init__(self, args, textData):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        # Placeholders
        self.encoderInputs  = None
        self.decoderInputs  = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # Construct the graphs
        self.buildNetwork()
 
    #To normalize the given dataset.
     def sampledSoftmax(self, labels, inputs):
            labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            localWt     = tf.cast(outputProjection.W_t, tf.float32)
            localB      = tf.cast(outputProjection.b, tf.float32)
            localInputs = tf.cast(inputs, tf.float32)

            return tf.cast(
                tf.nn.sampled_softmax_loss(
                    localWt,  # Should have shape [num_classes, dim]
                    localB,
                    labels,
                    localInputs,
                    self.args.softmaxSamples,  # The number of classes to randomly sample per batch
                    self.textData.getVocabularySize()),  # The number of classes
                self.dtype)

    def create_rnn_cell(self):
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(self.args.hiddenSize,)
            if not self.args.test:  # TODO: Should use a placeholder instead
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=self.args.dropout
                )
            return encoDecoCell

    def buildNetwork(self):
        #Create the computational graph

        # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)
        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():
            outputProjection = ProjectionOp(
                (self.textData.getVocabularySize(), self.args.hiddenSize),
                scope='softmax_projection',
                dtype=self.dtype
            )
        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLengthDeco)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
            self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
            encoDecoCell,
            self.textData.getVocabularySize(),
            self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
            embedding_size=self.args.embeddingSize,  # Dimension of each word
            output_projection=outputProjection.getWeights() if outputProjection else None,
            feed_previous=bool(self.args.test)  # When we test (self.args.test), we use previous output as next input (feed_previous)
        )

        # training and reduce memory usage. Other solution, use sampling softmax

        # For testing only
        if self.args.test:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]

        # For training only
        else:
            # Finally, we define the loss function
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function= sampledSoftmax if outputProjection else None  # If None, use default SoftMax
            )
            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
