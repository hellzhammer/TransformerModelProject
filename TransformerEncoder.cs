﻿public class TransformerEncoder
{
    private MultiHeadAttention _attention1;
    private MultiHeadAttention _attention2;
    private double[,] _W1, _W2;

    public TransformerEncoder(int embedDim, int numHeads)
    {
        _attention1 = new MultiHeadAttention(embedDim, numHeads);
        _attention2 = new MultiHeadAttention(embedDim, numHeads);
        _W1 = RandomMatrix(embedDim, embedDim * 2);
        _W2 = RandomMatrix(embedDim * 2, embedDim);
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

    public double[,] Forward(double[,] inputs)
    {
        var attnOutput1 = _attention1.ComputeAttention(inputs);
        var attnOutput2 = _attention2.ComputeAttention(attnOutput1);
        var hidden = MatrixUtils.Multiply(attnOutput2, _W1);
        var output = MatrixUtils.Multiply(hidden, _W2);
        return output;
    }
}
