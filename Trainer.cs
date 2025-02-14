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
                _decoder.Train(sample.Input, sample.ExpectedOutput, _embeddings, _tokenizer, _encoder, _decoder);
            }
        }
    }
}