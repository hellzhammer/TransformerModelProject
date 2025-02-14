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

    public static void UpdateWeightsAdam(double[,] weights, double[,] gradient, double[,] m, double[,] v, double learningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, int t = 1)
    {
        int rows = weights.GetLength(0);
        int cols = weights.GetLength(1);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                m[i, j] = beta1 * m[i, j] + (1 - beta1) * gradient[i, j];  // First moment
                v[i, j] = beta2 * v[i, j] + (1 - beta2) * gradient[i, j] * gradient[i, j];  // Second moment

                double mHat = m[i, j] / (1 - Math.Pow(beta1, t));  // Bias correction
                double vHat = v[i, j] / (1 - Math.Pow(beta2, t));

                weights[i, j] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
            }
    }

    public static void UpdateWeightsMomentum(double[,] weights, double[,] gradient, double[,] velocity, double learningRate, double momentum = 0.9)
    {
        int rows = weights.GetLength(0);
        int cols = weights.GetLength(1);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                velocity[i, j] = momentum * velocity[i, j] - learningRate * gradient[i, j]; // Update velocity
                weights[i, j] += velocity[i, j]; // Apply update
            }
    }

}