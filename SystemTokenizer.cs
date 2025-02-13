using TransformerTest.Models;

public class SystemTokenizer
{
    private Dictionary<string, int> _vocab = new Dictionary<string, int>();
    private int _nextTokenId = 1;

    public SystemTokenizer()
    {

    }

    public SystemTokenizer(Dictionary<string, int> vocab)
    {
        this._vocab = vocab;
    }

    private static int Current_Token_ID = 1;
    public static int GetNext()
    {
        var rtn = Current_Token_ID;
        Current_Token_ID++;
        return rtn;
    }

    /// <summary>
    /// builds a standard bag of words
    /// </summary>
    /// <param name="text"></param>
    /// <returns></returns>
    public int[] Tokenize(string text)
    {
        string[] words = text.ToLower().Split(' ');
        List<string> processed_words = new List<string>();
        for (int i = 0; i < words.Length; i++)
        {
            if (!string.IsNullOrWhiteSpace(words[i]))
            {
                processed_words.Add(words[i]);
            }
        }
        // go back to just using words to fix this.
        List<string> n_words = new List<string>();
        for (int i = 0; i < processed_words.Count - 1; i++)
        {
            n_words.Add(processed_words[i] + " " + processed_words[i + 1]);
        }

        int[] tokens = new int[n_words.Count];
        for (int i = 0; i < n_words.Count; i++)
        {
            if (!_vocab.ContainsKey(n_words[i]))
                _vocab[n_words[i]] = GetNext();
            tokens[i] = _vocab[n_words[i]];
        }

        return tokens;
    }

    public Dictionary<string, int> GetVocabulary()
    {
        return this._vocab;
    }

    public Dictionary<int, string> GetReverseTokenDict()
    {
        Dictionary<int, string> tokens = new Dictionary<int, string>();

        foreach (var word in _vocab)
        {
            tokens.Add(word.Value, word.Key);
        }

        return tokens;
    }
 
    /// <summary>
    /// to do. not really functional yet
    /// </summary>
    /// <param name="sentence"></param>
    /// <returns></returns>
    public List<string> ExtractBigrams(string sentence)
    {
        var words = sentence.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var bigrams = new List<string>();

        for (int i = 0; i < words.Length - 1; i++)
        {
            bigrams.Add(words[i] + " " + words[i + 1]);
        }

        return bigrams;
    }

    /*/// <summary>
    /// to do. not really functional yet
    /// </summary>
    /// <param name="sentence"></param>
    /// <returns></returns>
    public int[] ExtractBigrams(string[] sentence)
    {
        var words = sentence;
        //var bigrams = new List<string>();
        List<int> ints = new List<int>();

        for (int i = 0; i < words.Length - 1; i++)
        {
            if (_vocab.ContainsKey(words[i] + " " + words[i + 1]))
            {
                ints.Add();
            }
        }

        return //bigrams;
    }*/
}