using System.Text;

public class ResponseGenerator
{
    public string ConvertTokensToText(int[] tokens, Dictionary<int, string> vocab)
    {
        StringBuilder sb = new StringBuilder();

        foreach (int token in tokens)
        {
            if (vocab.ContainsKey(token))
            {
                string word = vocab[token];
                if (word == "s" || word == "e")
                    continue;  // Skip special tokens

                sb.Append(word + " ");
            }
        }

        return sb.ToString().Trim();
    }
}