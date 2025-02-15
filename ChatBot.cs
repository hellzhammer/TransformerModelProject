public class ChatBot
{
    private SystemTokenizer _tokenizer;
    private SystemEmbeddings _embeddings;
    private TransformerEncoder _encoder;
    private TransformerDecoder _decoder;
    private ResponseGenerator _responseGenerator;

    public ChatBot(SystemTokenizer systok, SystemEmbeddings sysemb, TransformerEncoder tranenc, TransformerDecoder trandec)
    {
        _tokenizer = systok;
        _embeddings = sysemb;
        _encoder = tranenc;
        _decoder = trandec;
        _responseGenerator = new ResponseGenerator();
    }

    public void Chat()
    {
        Console.WriteLine("Chatbot: Hello! Type 'exit' to quit.");
        while (true)
        {
            Console.Write("You: ");
            string input = Console.ReadLine();
            if (input.ToLower() == "exit" || input.ToLower() == "quit")
                break;
            string n_input = string.Empty;
            var filtered = NGram_Generator.Pad_Filter(input.ToLower().Split(' '), true, true);
            for (int i = 0; i < filtered.Length; i++)
            {
                if (i == 0)
                {
                    n_input += filtered[i];
                }
                else
                {
                    n_input += " " + filtered[i];
                }
            }

            int[] tokens = _tokenizer.Tokenize(n_input.ToLower());
            double[,] embedded = _embeddings.GetEmbedding(tokens);
            double[,] encoded = _encoder.Forward(embedded);
            double[,] decoded = _decoder.Forward(encoded, embedded);
            var logits = MatrixUtils.Softmax(decoded);
            var textTokens = MatrixUtils.DecodeTokens(logits);
            Console.WriteLine();

            Console.Write("Thinking: ");
            for (int i = 0; i < textTokens.Length; i++)
            {
                Console.Write(textTokens[i] + " ");
            }

            Console.WriteLine();
            var resp = _responseGenerator.ConvertTokensToText(textTokens, _tokenizer.GetReverseTokenDict());

            Console.WriteLine($"Chatbot Process: {resp}");
        }
    }
}