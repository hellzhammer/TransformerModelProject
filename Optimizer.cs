public static class Optimizer
{
    public static void UpdateWeights(double[,] weights, double[,] gradient, double learningRate)
    {
        int rows = weights.GetLength(0); 
        int cols = weights.GetLength(1);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                weights[i, j] -= learningRate * gradient[i, j]; // Gradient Descent
    }
}