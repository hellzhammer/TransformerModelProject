public class TrainableTransformerDecoder
{
    private MultiHeadAttention _selfAttention;
    private MultiHeadAttention _crossAttention;
    private double[,] _W1, _W2;
    private double _learningRate = 0.001;

    public TrainableTransformerDecoder(int embedDim, int numHeads)
    {
        _selfAttention = new MultiHeadAttention(embedDim, numHeads);
        _crossAttention = new MultiHeadAttention(embedDim, numHeads);
        _W1 = XavierMatrix(embedDim, embedDim * 2);
        _W2 = XavierMatrix(embedDim * 2, embedDim);
    }

    public TransformerDecoder GetDecoder()
    {
        return new TransformerDecoder(this._selfAttention, this._crossAttention, _W1, _W2);
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
        var selfAttnOutput = _selfAttention.ComputeAttention(decoderInput);
        var crossAttnOutput = _crossAttention.ComputeAttention(selfAttnOutput, encoderOutput, encoderOutput);

        var hidden = MatrixUtils.Multiply(crossAttnOutput, _W1);
        var output = MatrixUtils.Multiply(hidden, _W2);

        // Ensure output shape matches expectedOutput
        output = MatrixUtils.Resize(output, expectedRows, expectedCols);

        return output;
    }

    public void Train(double[,] encoderOutput, double[,] decoderInput, double[,] expectedOutput)
    {
        int expRows = expectedOutput.GetLength(0);
        int expCols = expectedOutput.GetLength(1);
        var predictedOutput = Forward(encoderOutput, decoderInput, expRows, expCols);
        Console.WriteLine($"Predicted: {predictedOutput.GetLength(0)}x{predictedOutput.GetLength(1)}, Expected: {expectedOutput.GetLength(0)}x{expectedOutput.GetLength(1)}");
        var loss = LossFunction.ComputeLoss(predictedOutput, expectedOutput);

        // Compute the gradient for the final output
        var outputError = LossFunction.ComputeGradient(predictedOutput, expectedOutput, _W2.GetLength(1)); // [batchSize, embedDim]

        // Compute the correct gradient for W2
        var atten = _crossAttention.ComputeAttention(encoderOutput);
        var hiddenLayerOutput = MatrixUtils.Multiply(atten, _W1); 
        var hiddenlayertranspose = MatrixUtils.Transpose(hiddenLayerOutput);
        var gradient_W2 = MatrixUtils.Multiply(hiddenlayertranspose, outputError); 

        // back propogate
        var w2_transpose = MatrixUtils.Transpose(_W2);
        var hiddenGradient = MatrixUtils.Multiply(gradient_W2, w2_transpose); 
        var encoder_transpose = MatrixUtils.Transpose(encoderOutput);
        var gradient_W1 = MatrixUtils.Multiply(encoder_transpose, hiddenGradient); 

        // apply clipping
        gradient_W1 = LossFunction.ClipGradients(gradient_W1, 10.0);
        gradient_W2 = LossFunction.ClipGradients(gradient_W2, 10.0);

        // Update weights using the gradient
        Optimizer.UpdateWeights(_W1, gradient_W1, _learningRate);
        Optimizer.UpdateWeights(_W2, gradient_W2, _learningRate);

        Console.WriteLine($"Current Loss: {loss}");
    }
}