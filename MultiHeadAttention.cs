public class MultiHeadAttention
{
    private int _embedDim, _numHeads, _headDim;
    private double[,] _Wq, _Wk, _Wv, _Wo;

    public MultiHeadAttention(int embedDim, int numHeads)
    {
        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads; // Split the embedding size across heads

        _Wq = XavierMatrix(embedDim, embedDim);
        _Wk = XavierMatrix(embedDim, embedDim);
        _Wv = XavierMatrix(embedDim, embedDim);
        _Wo = XavierMatrix(embedDim, embedDim); // Output projection
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
        // Step 1: Linear Projections
        var Q = MatrixUtils.Multiply(inputs, _Wq);
        var K = MatrixUtils.Multiply(inputs, _Wk);
        var V = MatrixUtils.Multiply(inputs, _Wv);

        // Step 2: Split into multiple heads
        var Q_heads = SplitIntoHeads(Q);
        var K_heads = SplitIntoHeads(K);
        var V_heads = SplitIntoHeads(V);

        // Step 3: Compute attention for each head
        var attentionOutputs = new double[_numHeads][,];
        for (int h = 0; h < _numHeads; h++)
        {
            var scores = MatrixUtils.Multiply(Q_heads[h], MatrixUtils.Transpose(K_heads[h]));
            MatrixUtils.Scale(scores, 1.0 / Math.Sqrt(_headDim)); // Scale by sqrt(d_k)
            var attentionWeights = MatrixUtils.Softmax(scores);
            attentionOutputs[h] = MatrixUtils.Multiply(attentionWeights, V_heads[h]);
        }

        // Step 4: Concatenate heads & project
        var concatenated = ConcatenateHeads(attentionOutputs);
        return MatrixUtils.Multiply(concatenated, _Wo); // Apply output projection
    }

    private double[][,] SplitIntoHeads(double[,] input)
    {
        int seqLen = input.GetLength(0);
        double[][,] heads = new double[_numHeads][,];

        for (int h = 0; h < _numHeads; h++)
        {
            heads[h] = new double[seqLen, _headDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < _headDim; j++)
                    heads[h][i, j] = input[i, h * _headDim + j]; // Split dimensions per head
        }
        return heads;
    }

    private double[,] ConcatenateHeads(double[][,] heads)
    {
        int seqLen = heads[0].GetLength(0);
        double[,] concatenated = new double[seqLen, _embedDim];

        for (int h = 0; h < _numHeads; h++)
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < _headDim; j++)
                    concatenated[i, h * _headDim + j] = heads[h][i, j]; // Merge heads

        return concatenated;
    }
}