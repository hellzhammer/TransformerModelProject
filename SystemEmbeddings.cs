using System;

public class SystemEmbeddings
{
    private double[,] _embeddingMatrix;
    private int _vocabSize, _embedDim;
    private const int UNK_TOKEN = 0; // Special Unknown Token

    public SystemEmbeddings(int vocabSize, int embedDim)
    {
        _vocabSize = vocabSize;
        _embedDim = embedDim;
        _embeddingMatrix = new double[vocabSize + 1, embedDim]; // +1 for unknown token

        Random rand = new Random();

        // Gaussian Initialization: Mean 0, Std 1/sqrt(embedDim)
        double stddev = 1.0 / Math.Sqrt(embedDim);
        for (int i = 0; i <= vocabSize; i++)
            for (int j = 0; j < embedDim; j++)
                _embeddingMatrix[i, j] = GaussianRandom(rand, 0, stddev);
    }

    private double GaussianRandom(Random rand, double mean, double stddev)
    {
        // Box-Muller Transform to generate normal distribution
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stddev * normal;
    }

    public double[,] GetEmbedding(int[] tokens)
    {
        int seqLen = tokens.Length;
        double[,] result = new double[seqLen, _embedDim];

        for (int i = 0; i < seqLen; i++)
        {
            int tokenIndex = (tokens[i] >= 0 && tokens[i] <= _vocabSize) ? tokens[i] : UNK_TOKEN;
            for (int j = 0; j < _embedDim; j++)
                result[i, j] = _embeddingMatrix[tokenIndex, j];
        }

        return result;
    }
}


/*public class SystemEmbeddings
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
}*/