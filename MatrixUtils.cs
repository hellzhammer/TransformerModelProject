public static class MatrixUtils
{
    public static double[,] Multiply(double[,] a, double[,] b)
    {
        int m = a.GetLength(0), n = a.GetLength(1), p = b.GetLength(1);
        double[,] result = new double[m, p];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < p; j++)
                for (int k = 0; k < n; k++)
                    result[i, j] += a[i, k] * b[k, j];
        return result;
    }

    public static double[,] Resize(double[,] input, int newRows, int newCols)
    {
        int oldRows = input.GetLength(0), oldCols = input.GetLength(1);
        double[,] resized = new double[newRows, newCols];

        for (int i = 0; i < Math.Min(oldRows, newRows); i++)
            for (int j = 0; j < Math.Min(oldCols, newCols); j++)
                resized[i, j] = input[i, j]; // Copy existing values

        return resized;
    }


    public static double[,] Transpose(double[,] matrix)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        double[,] result = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j, i] = matrix[i, j];
        return result;
    }

    // Scale Matrix Elements (For Attention Scores)
    public static void Scale(double[,] matrix, double factor)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] *= factor;
    }

    // Debugging: Print Matrix
    public static void PrintMatrix(string label, double[,] matrix)
    {
        Console.WriteLine(label);
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
                Console.Write($"{matrix[i, j]:0.000} ");
            Console.WriteLine();
        }
        Console.WriteLine();
    }

    public static double[,] Softmax(double[,] scores)
    {
        int rows = scores.GetLength(0), cols = scores.GetLength(1);
        double[,] softmax = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            double maxVal = double.NegativeInfinity;
            for (int j = 0; j < cols; j++)
                maxVal = Math.Max(maxVal, scores[i, j]); // Normalize

            double sum = 0.0;
            for (int j = 0; j < cols; j++)
            {
                softmax[i, j] = Math.Exp(scores[i, j] - maxVal); // Subtract max to prevent overflow
                sum += softmax[i, j];
            }

            if (sum == 0 || double.IsNaN(sum)) sum = 1; // Prevent division by zero

            for (int j = 0; j < cols; j++)
                softmax[i, j] /= sum;
        }

        return softmax;
    }

    public static int[] DecodeTokens(double[,] probabilities, int k = 5)
    {
        int sequenceLength = probabilities.GetLength(0);
        int vocabSize = probabilities.GetLength(1);
        int[] decodedTokens = new int[sequenceLength];

        for (int i = 0; i < sequenceLength; i++)
        {
            double[] rowProbs = new double[vocabSize];
            for (int j = 0; j < vocabSize; j++)
                rowProbs[j] = probabilities[i, j];

            decodedTokens[i] = SampleFromTopK(rowProbs, k);
        }

        return decodedTokens;
    }

    private static int SampleFromTopK(double[] probabilities, int k = 5)
    {
        int vocabSize = probabilities.Length;
        List<(int, double)> sortedProbs = new List<(int, double)>();

        for (int i = 0; i < vocabSize; i++)
            sortedProbs.Add((i, probabilities[i]));

        // Sort by probability (descending) and take the top k
        sortedProbs = sortedProbs.OrderByDescending(x => x.Item2).Take(k).ToList();

        // Normalize top-k probabilities
        double sum = sortedProbs.Sum(x => x.Item2);
        for (int i = 0; i < sortedProbs.Count; i++)
            sortedProbs[i] = (sortedProbs[i].Item1, sortedProbs[i].Item2 / sum);

        // Sample from the top-k probabilities
        double rand = new Random().NextDouble();
        double cumulative = 0.0;

        foreach (var (token, prob) in sortedProbs)
        {
            cumulative += prob;
            if (rand < cumulative)
                return token;
        }

        return sortedProbs.Last().Item1; // Fallback
    }

    public static bool ContainsNaN(double[,] matrix)
    {
        foreach (double value in matrix)
            if (double.IsNaN(value)) return true;
        return false;
    }
}
