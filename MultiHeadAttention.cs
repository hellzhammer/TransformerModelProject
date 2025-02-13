public class MultiHeadAttention
{
    private int _embedDim, _numHeads;
    private double[,] _Wq, _Wk, _Wv, _Wo;

    public MultiHeadAttention(int embedDim, int numHeads)
    {
        _embedDim = embedDim;
        _numHeads = numHeads;
        _Wq = XavierMatrix(embedDim, embedDim);
        _Wk = XavierMatrix(embedDim, embedDim);
        _Wv = XavierMatrix(embedDim, embedDim);
        _Wo = XavierMatrix(embedDim, embedDim);
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

    public double[,] ComputeAttention(double[,] inputs)
    {
        var Q = MatrixUtils.Multiply(inputs, _Wq);
        var K = MatrixUtils.Multiply(inputs, _Wk);
        var V = MatrixUtils.Multiply(inputs, _Wv);

        var K_T = MatrixUtils.Transpose(K);
        var attentionScores = MatrixUtils.Multiply(Q, K_T);
        var attentionWeights = MatrixUtils.Softmax(attentionScores);
        return MatrixUtils.Multiply(attentionWeights, V);
    }

    public double[,] ComputeAttention(double[,] query, double[,] key, double[,] value)
    {
        // Q * K^T / sqrt(d_k)
        var scores = MatrixUtils.Multiply(query, MatrixUtils.Transpose(key));
        MatrixUtils.Scale(scores, 1.0 / Math.Sqrt(query.GetLength(1)));

        var attentionWeights = MatrixUtils.Softmax(scores);

        return MatrixUtils.Multiply(attentionWeights, value);
    }
}
