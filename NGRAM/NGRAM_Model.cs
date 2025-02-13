using TransformerTest.Models;

public class NGram_Generator
{
    // private readonly list of approved characters -- alphabet and numerals.
    private static readonly List<char> alpha_numeric_chars = new List<char>
        {
            'a','b','c','d','e','f','g','h','i','j',
            'k','l','m','n','o','p','q','r','s','t',
            'u','v','w','x','y','z','0','1','2','3',
            '4','5','6','7','8','9'
        };

    public static string[] Pad_Filter(string[] raw, bool pad_left, bool pad_right)
    {
        for (int r = 0; r < raw.Length; r++)
        {
            if (!string.IsNullOrWhiteSpace(raw[r]))
            {
                // pad and filter text data
                string filtered = string.Empty;
                raw[r] = raw[r].ToLower();
                if (pad_left)
                {
                    filtered += "s ";
                }
                for (int i = 0; i < raw[r].Length; i++)
                {
                    filtered += "";
                    if (!alpha_numeric_chars.Contains(raw[r][i]) && raw[r][i] != ' ')
                    {
                        if (raw[r][i] == '/' || raw[r][i] == '-')
                        {
                            if (i < raw[r].Length - 1 && i > 1)
                            {
                                if (alpha_numeric_chars.Contains(raw[r][i - 1])
                                    ||
                                    alpha_numeric_chars.Contains(raw[r][i + 1]))
                                {
                                    filtered += " ";
                                }
                            }
                        }
                    }
                    else
                    {
                        filtered += raw[r][i];
                    }
                }
                if (pad_right)
                {
                    filtered += " e";
                }

                raw[r] = filtered;
            }
        }

        return raw;
    }

    public static string Combine(string[] raw)
    {
        string rtnval = raw[0];
        for (int r = 0; r < raw.Length; r++)
        {
            rtnval += " " + raw[r];
        }

        return rtnval;
    }

    public static NGram_Model<string[]> Generate(string[] raw, int n)
    {
        int nID = 0;
        List<string> tokens_in_order = new List<string>();
        Dictionary<string, int> n_grams = new Dictionary<string, int>();
        Dictionary<string, int> token_dict = new Dictionary<string, int>();
        Dictionary<string, int> n_minus_one = new Dictionary<string, int>();

        for (int r = 0; r < raw.Length; r++)
        {
            if (!string.IsNullOrWhiteSpace(raw[r]))
            {
                // remove all excess white space
                var split = raw[r].Split(" ");
                List<string> _split = new List<string>();
                for (int i = 0; i < split.Length; i++)
                {
                    if (!string.IsNullOrWhiteSpace(split[i]))
                    {
                        _split.Add(split[i]);
                        tokens_in_order.Add(split[i]);
                    }
                }
                split = _split.ToArray();

                // tokenize / extract unigram dictionary 
                for (int i = 0; i < split.Length; i++)
                {
                    if (!token_dict.ContainsKey(split[i])
                        &&
                        !string.IsNullOrWhiteSpace(split[i]))
                    {
                        token_dict.Add(split[i], 1);
                    }
                    else
                    {
                        token_dict[split[i]]++;
                    }
                }

                // n gram dictionary
                for (int i = 0; i < split.Length; i++)
                {
                    if (i < split.Length - 1)
                    {
                        string[] tok = new string[n];
                        for (int j = 0; j < n; j++)
                        {
                            if (i + j <= split.Length - 1)
                            {
                                tok[j] = split[i + j];
                            }
                        }
                        var key = tok[0];
                        for (int y = 1; y < tok.Length; y++)
                        {
                            key += " " + tok[y];
                        }

                        bool filter = false;
                        for (int j = 0; j < tok.Length; j++)
                        {
                            if (string.IsNullOrWhiteSpace(tok[j]))
                            {
                                filter = true;
                                break;
                            }
                        }

                        if (!filter)
                        {
                            if (!n_grams.ContainsKey(key) && tok[1] != null)
                            {
                                n_grams.Add(key, nID);
                                nID++;
                            }
                        }
                    }
                }
            }
        }

        return new NGram_Model<string[]>()
        {
            raw_text = raw,
            split_text = tokens_in_order.ToArray(),
            n_gram_count = n,
            n_grams = n_grams,
            vocabulary_dict = token_dict
        };
    }

    public static double[][] Generate(double[] raw, int n)
    {
        List<double[]> rtn = new List<double[]>();

        for (int r = 0; r < raw.Length; r++)
        {
            // n gram dictionary
            List<double> col = new List<double>();
            if (r + (n + 1) <= raw.Length)
            {
                for (int i = r; i < r + (n + 1); i++)
                {
                    col.Add(raw[i]);
                }
            }
            if (col.Count > 0 && col.Count == n + 1)
            {
                rtn.Add(col.ToArray());
            }
        }

        return rtn.ToArray();
    }

    public string[] remove_unwanted_language(string[] original, List<string> words_to_remove)
    {
        List<string> _filtered = new List<string>();

        for (int i = 0; i < original.Length; i++)
        {
            if (!words_to_remove.Contains(original[i]))
            {
                _filtered.Add(original[i]);
            }
        }

        return _filtered.ToArray();
    }

    public static (List<string>, List<double>) Bigram(string[] tokens, NGram_Model<string[]> model)
    {
        var model2 = new NGram_Model<string>
        {
            n_grams = model.n_grams,
            n_gram_count = model.n_gram_count,
            split_text = model.split_text,
            vocabulary_dict = model.vocabulary_dict
        };

        var outp = Bigram(tokens, model2);
        return outp;
    }

    private static (List<string>, List<double>) Bigram(string[] tokens, NGram_Model<string> model)
    {
        List<string> items = new List<string>();
        List<double> items2 = new List<double>();

        foreach (var tok in model.n_grams.Keys)
        {
            var spl = tok.Split(' ');
            if (spl[0] == tokens[0])
            {
                tokens[tokens.Length - 1] = spl[tokens.Length - 1];
                var key = tokens[0] + " " + tokens[tokens.Length - 1];
                items.Add(key);
                var one = model.n_grams.ContainsKey(key);
                var two = model.vocabulary_dict.ContainsKey(tokens[0]);
                if (one && two)
                {
                    double outp = (double)model.n_grams[key] / (double)model.vocabulary_dict[tokens[0]];
                    items2.Add(outp);
                }
            }
        }

        return (items, items2);
    }
}