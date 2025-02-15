public class TrainableTransformerDecoder
{
    private MultiHeadAttention _selfAttention;
    private MultiHeadAttention _crossAttention;
    private double[,] _W1, _W2;
    private double[,] _mW1, _vW1, _mW2, _vW2;
    private double _learningRate = 0.0001;
    private double _beta1 = 0.9, _beta2 = 0.999, _epsilon = 1e-8;
    private int _t = 1;

    public TrainableTransformerDecoder(int embedDim, int numHeads)
    {
        _selfAttention = new MultiHeadAttention(embedDim, numHeads);
        _crossAttention = new MultiHeadAttention(embedDim, numHeads);
        _W1 = XavierMatrix(embedDim, embedDim * 2);
        _W2 = XavierMatrix(embedDim * 2, embedDim);

        // Initialize Adam moment estimates
        _mW1 = new double[embedDim, embedDim * 2];
        _vW1 = new double[embedDim, embedDim * 2];
        _mW2 = new double[embedDim * 2, embedDim];
        _vW2 = new double[embedDim * 2, embedDim];
    }

    public TransformerDecoder GetDecoder()
    {
        return new TransformerDecoder(_selfAttention, _crossAttention, _W1, _W2);
    }

    private double[,] XavierMatrix(int rows, int cols)
    {
        Random rand = new Random();
        double scale = Math.Sqrt(2.0 / (rows + cols));
        double[,] matrix = new double[rows, cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = (rand.NextDouble() * 2 - 1) * scale;

        return matrix;
    }

    public void Train(string input, string expectedResponse, SystemEmbeddings _embeddings, SystemTokenizer _tokenizer, TransformerEncoder _encoder)
    {
        // Tokenize input and expected response
        int[] inputTokens = _tokenizer.Tokenize(input.ToLower());
        int[] expectedTokens = _tokenizer.Tokenize(expectedResponse.ToLower());

        // Convert tokens to embeddings
        double[,] embeddedInput = _embeddings.GetEmbedding(inputTokens);
        double[,] embeddedExpected = _embeddings.GetEmbedding(expectedTokens);

        // Forward pass through encoder & decoder
        double[,] encoded = _encoder.Forward(embeddedInput);
        double[,] decoded = Forward(encoded, embeddedInput, embeddedExpected.GetLength(0), embeddedExpected.GetLength(1));

        // Compute softmax and loss
        double[,] logits = MatrixUtils.Softmax(decoded);
        var loss = LossFunction.ComputeLoss(logits, embeddedExpected);
        Console.WriteLine($"Current Loss: {loss}");

        // Compute gradient for output layer
        var outputError = LossFunction.ComputeGradient(logits, embeddedExpected, _W2.GetLength(1));
        var attention = _crossAttention.ComputeAttention(encoded);
        var hiddenLayerOutput = MatrixUtils.Multiply(attention, _W1);
        var hiddenLayerTranspose = MatrixUtils.Transpose(hiddenLayerOutput);

        // Compute gradients for weights
        var gradient_W2 = MatrixUtils.Multiply(hiddenLayerTranspose, outputError);
        var w2_transpose = MatrixUtils.Transpose(_W2);
        var hiddenGradient = MatrixUtils.Multiply(outputError, w2_transpose);
        var encoder_transpose = MatrixUtils.Transpose(encoded);
        var gradient_W1 = MatrixUtils.Multiply(encoder_transpose, hiddenGradient);

        // Apply gradient clipping
        //gradient_W1 = LossFunction.ClipGradients(gradient_W1, 5);
        //gradient_W2 = LossFunction.ClipGradients(gradient_W2, 5);

        // Update weights using Adam optimizer
        Optimizer.UpdateWeightsAdam(_W1, gradient_W1, _mW1, _vW1, _learningRate, _beta1, _beta2, _epsilon, _t);
        Optimizer.UpdateWeightsAdam(_W2, gradient_W2, _mW2, _vW2, _learningRate, _beta1, _beta2, _epsilon, _t);

        _t++; // Increment time step
    }

    public double[,] Forward(double[,] encoderOutput, double[,] decoderInput, int expectedRows, int expectedCols)
    {
        // Self-Attention: Processes decoder input (for autoregressive behavior)
        var selfAttnOutput = _selfAttention.ComputeAttention(decoderInput);

        // Cross-Attention: Links encoder and decoder representations
        var crossAttnOutput = _crossAttention.ComputeAttention(encoderOutput);

        // Combine both attention outputs (concatenation or element-wise sum)
        var combinedAttn = MatrixUtils.Add(selfAttnOutput, crossAttnOutput);

        // Hidden Layer Transformation
        var hidden = MatrixUtils.Multiply(combinedAttn, _W1);

        // Output Transformation
        var output = MatrixUtils.Multiply(hidden, _W2);

        // Ensure output shape matches expectedOutput
        output = MatrixUtils.Resize(output, expectedRows, expectedCols);

        return output;
    }
}


/*public class TrainableTransformerDecoder
{
    private MultiHeadAttention _selfAttention;
    private MultiHeadAttention _crossAttention;
    private double[,] _W1, _W2;
    private double _learningRate = 0.0001;

    public TrainableTransformerDecoder(int embedDim, int numHeads)
    {
        _selfAttention = new MultiHeadAttention(embedDim, numHeads);
        _crossAttention = new MultiHeadAttention(embedDim, numHeads);
        _W1 = XavierMatrix(embedDim, embedDim * 2);
        _W2 = XavierMatrix(embedDim * 2, embedDim);
    }

    public TransformerDecoder GetDecoder()
    {
        return new TransformerDecoder(_selfAttention, _crossAttention, _W1, _W2);
    }

    private double[,] RandomMatrix(int rows, int cols)
    {
        Random rand = new Random();
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = rand.NextDouble();
        return matrix;
    }

    private double[,] XavierMatrix(int rows, int cols)
    {
        Random rand = new Random();
        double scale = Math.Sqrt(2.0 / (rows + cols));
        double[,] matrix = new double[rows, cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = (rand.NextDouble() * 2 - 1) * scale;

        return matrix;
    }

    public double[,] Forward(double[,] encoderOutput, double[,] decoderInput, int expectedRows, int expectedCols)
    {
        // Self-Attention: Processes decoder input (for autoregressive behavior)
        var selfAttnOutput = _selfAttention.ComputeAttention(decoderInput);

        // Cross-Attention: Links encoder and decoder representations
        var crossAttnOutput = _crossAttention.ComputeAttention(encoderOutput);

        // Combine both attention outputs (concatenation or element-wise sum)
        var combinedAttn = MatrixUtils.Add(selfAttnOutput, crossAttnOutput);

        // Hidden Layer Transformation
        var hidden = MatrixUtils.Multiply(combinedAttn, _W1);

        // Output Transformation
        var output = MatrixUtils.Multiply(hidden, _W2);

        // Ensure output shape matches expectedOutput
        output = MatrixUtils.Resize(output, expectedRows, expectedCols);

        return output;
    }

    public void Train(string input, string expectedResponse, SystemEmbeddings _embeddings, SystemTokenizer _tokenizer, TransformerEncoder _encoder, TrainableTransformerDecoder _decoder)
    {
        // Tokenize input and expected response
        int[] inputTokens = _tokenizer.Tokenize(input.ToLower());
        int[] expectedTokens = _tokenizer.Tokenize(expectedResponse.ToLower());

        // Convert tokens to embeddings
        double[,] embeddedInput = _embeddings.GetEmbedding(inputTokens);
        double[,] embeddedExpected = _embeddings.GetEmbedding(expectedTokens);

        // Get expected output dimensions
        int rows = embeddedExpected.GetLength(0);
        int cols = embeddedExpected.GetLength(1);

        // Forward pass through the encoder & decoder (same as Chat function)
        double[,] encoded = _encoder.Forward(embeddedInput);
        double[,] decoded = _decoder.Forward(encoded, embeddedInput, rows, cols); // Now includes expected shape

        // Compute softmax to get probabilities
        double[,] logits = MatrixUtils.Softmax(decoded);

        // Compute loss
        var loss = LossFunction.ComputeLoss(logits, embeddedExpected);
        Console.WriteLine($"Current Loss: {loss}");

        // Compute the gradient for the output layer
        var outputError = LossFunction.ComputeGradient(logits, embeddedExpected, _W2.GetLength(1));

        // Compute attention (same as in Chat)
        var attention = _crossAttention.ComputeAttention(encoded);

        // Compute hidden layer output
        var hiddenLayerOutput = MatrixUtils.Multiply(attention, _W1);
        var hiddenLayerTranspose = MatrixUtils.Transpose(hiddenLayerOutput);

        // Compute gradients for weights
        var gradient_W2 = MatrixUtils.Multiply(hiddenLayerTranspose, outputError);
        var w2_transpose = MatrixUtils.Transpose(_W2);
        var hiddenGradient = MatrixUtils.Multiply(gradient_W2, w2_transpose);
        var encoder_transpose = MatrixUtils.Transpose(encoded);
        var gradient_W1 = MatrixUtils.Multiply(encoder_transpose, hiddenGradient);

        // Apply gradient clipping
        gradient_W1 = LossFunction.ClipGradients(gradient_W1, 5);
        gradient_W2 = LossFunction.ClipGradients(gradient_W2, 5);

        // Update weights
        Optimizer.UpdateWeights(_W1, gradient_W1, _learningRate);
        Optimizer.UpdateWeights(_W2, gradient_W2, _learningRate);
    }
}*/