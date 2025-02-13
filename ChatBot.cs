using System.Security.Cryptography.X509Certificates;

public class ChatBot
{
    private SystemTokenizer _tokenizer;
    private SystemEmbeddings _embeddings;
    private TransformerEncoder _encoder;
    private TransformerDecoder _decoder;
    private ResponseGenerator _responseGenerator;

    public ChatBot()
    {
        _tokenizer = new SystemTokenizer();
        _embeddings = new SystemEmbeddings(1000, 16);
        _encoder = new TransformerEncoder(16, 2);
        _decoder = new TransformerDecoder(16, 2);
        _responseGenerator = new ResponseGenerator();
    }

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
            string current_response = string.Empty;
            Console.Write("You: ");
            string input = Console.ReadLine();
            if (input.ToLower() == "exit") break;

            /*for (int j = 0; j < 10; j++)
            {*/
                int[] tokens = _tokenizer.Tokenize(input);
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
                var resp = _responseGenerator.ConvertTokensToText(textTokens, _tokenizer.GetReverseTokenDict()); // change this.

                var _res = resp.Split(' ')[0];
                current_response += " " + _res;
                input = input + " " + _res;

                Console.WriteLine($"Chatbot Process: {resp}");
            //}

            Console.WriteLine($"Chatbot: {current_response}");
        }
    }
}
