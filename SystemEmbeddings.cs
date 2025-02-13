public class SystemEmbeddings
{
    private double[,] _embeddingMatrix;

    public SystemEmbeddings(int vocabSize, int embedDim)
    {
        _embeddingMatrix = new double[vocabSize + 1, embedDim]; // +1 for unknown tokens
        Random rand = new Random();
        for (int i = 0; i <= vocabSize; i++)
            for (int j = 0; j < embedDim; j++)
                _embeddingMatrix[i, j] = rand.NextDouble();
    }

    public double[,] GetEmbedding(int[] tokens)
    {
        int seqLen = tokens.Length, embedDim = _embeddingMatrix.GetLength(1);
        double[,] result = new double[seqLen, embedDim];
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < embedDim; j++)
                result[i, j] = _embeddingMatrix[tokens[i], j];
        return result;
    }
}