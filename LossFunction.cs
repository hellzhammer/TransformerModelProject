public static class LossFunction
{
    public static double ComputeLoss(double[,] predicted, double[,] expected)
    {
        int rowsPred = predicted.GetLength(0), colsPred = predicted.GetLength(1);
        int rowsExp = expected.GetLength(0), colsExp = expected.GetLength(1);

        // Ensure they have the same shape before loss calculation
        if (rowsPred != rowsExp || colsPred != colsExp)
        {
            Console.WriteLine($"Warning: Mismatched shapes! Resizing predicted [{rowsPred}x{colsPred}] to expected [{rowsExp}x{colsExp}].");
            predicted = MatrixUtils.Resize(predicted, rowsExp, colsExp);
        }

        double sum = 0.0;

        for (int i = 0; i < rowsExp; i++)
            for (int j = 0; j < colsExp; j++)
            {
                double diff = predicted[i, j] - expected[i, j];
                sum += diff * diff;
            }

        return sum / Math.Max(1, rowsExp * colsExp);
    }

    public static double[,] ComputeGradient(double[,] predicted, double[,] expected, int weightCols)
    {
        int rows = predicted.GetLength(0); 
        int cols = predicted.GetLength(1); 

        if (rows == 0 || cols == 0 || weightCols == 0)
            throw new InvalidOperationException("Gradient shape cannot be zero");

        double[,] gradient = new double[cols, weightCols];

        for (int i = 0; i < cols; i++)
        {
            double diff = predicted[0, i] - expected[0, i];

            if (double.IsNaN(diff) || double.IsInfinity(diff))
                diff = 0; // Prevent NaNs

            for (int j = 0; j < weightCols; j++)
                gradient[i, j] = 2 * diff / Math.Max(1, (rows * cols)); // Avoid zero division
        }

        return gradient;
    }

    /// <summary>
    /// prevents the gradient from exploding. usually.
    /// </summary>
    public static double[,] ClipGradients(double[,] gradient, double threshold)
    {
        int rows = gradient.GetLength(0), cols = gradient.GetLength(1);
        double maxVal = double.NegativeInfinity;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                maxVal = Math.Max(maxVal, Math.Abs(gradient[i, j]));

        if (maxVal > threshold)
        {
            double scale = threshold / maxVal;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    gradient[i, j] *= scale;
        }
        return gradient;
    }
}
