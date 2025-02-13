public class Trainer
{
    private SystemTokenizer _tokenizer;
    private SystemEmbeddings _embeddings;
    private TransformerEncoder _encoder;
    private TrainableTransformerDecoder _decoder;

    private TrainingData[] training_data { get; set; }

    public Trainer(TrainingData[] data, Dictionary<string, int> vocab)
    {
        this.training_data = data;
        _tokenizer = new SystemTokenizer(vocab);
        _embeddings = new SystemEmbeddings(10000, 30);
        _encoder = new TransformerEncoder(30, 5);
        _decoder = new TrainableTransformerDecoder(30, 5);
    }

    public ChatBot CreateNewBot()
    {
        ChatBot cb = new ChatBot(_tokenizer, _embeddings, _encoder, _decoder.GetDecoder());
        return cb;
    }

    public void TrainModel(int epochs)
    {
        var dataset = this.training_data;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}");

            foreach (var sample in dataset)
            {
                Console.WriteLine("Epoch: " + epoch);
                int[] inputTokens = _tokenizer.Tokenize(sample.Input);
                int[] expectedTokens = _tokenizer.Tokenize(sample.ExpectedOutput);

                double[,] inputEmbeddings = _embeddings.GetEmbedding(inputTokens);
                double[,] expectedEmbeddings = _embeddings.GetEmbedding(expectedTokens);

                double[,] encoderOutput = _encoder.Forward(inputEmbeddings);
                _decoder.Train(encoderOutput, inputEmbeddings, expectedEmbeddings);
            }
        }
    }
}