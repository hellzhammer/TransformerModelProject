public class TransformerDecoder
{
    private MultiHeadAttention _selfAttention;
    private MultiHeadAttention _crossAttention1;
    private double[,] _W1, _W2; // Feed-forward network weights

    public TransformerDecoder(int embedDim, int numHeads)
    {
        _selfAttention = new MultiHeadAttention(embedDim, numHeads);
        _crossAttention1 = new MultiHeadAttention(embedDim, numHeads);
        _W1 = RandomMatrix(embedDim, embedDim * 2);
        _W2 = RandomMatrix(embedDim * 2, embedDim);
    }

    public TransformerDecoder(MultiHeadAttention h1, MultiHeadAttention h2, double[,] w1, double[,] w2)
    {
        _selfAttention = h1;
        _crossAttention1 = h2;
        _W1 = w1;
        _W2 = w2;
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

    public double[,] Forward(double[,] encoderOutput, double[,] decoderInput)
    {
        // Self-attention on decoder input
        var selfAttnOutput = _selfAttention.ComputeAttention(decoderInput);

        // First cross-attention using encoder output as context
        var crossAttnOutput1 = _crossAttention1.ComputeAttention(encoderOutput);

        // Combine self-attention and cross-attention outputs
        var combined = MatrixUtils.Add(selfAttnOutput, crossAttnOutput1);

        // Feed-forward network
        var hidden = MatrixUtils.Multiply(combined, _W1);
        var output = MatrixUtils.Multiply(hidden, _W2);

        return output;
    }
}
