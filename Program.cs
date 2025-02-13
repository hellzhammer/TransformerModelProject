using TransformerTest.Models;

public class Program
{
    public static void Main()
    {
        var d = GetData();

        Dictionary<string, int> nvalues = new Dictionary<string, int>();

        foreach (var item in d.Item2.n_grams)
        {
            if (!nvalues.ContainsKey(item.Key))
            {
                nvalues.Add(item.Key, SystemTokenizer.GetNext());
            }
        }

        foreach (var item in d.Item3.n_grams)
        {
            if (!nvalues.ContainsKey(item.Key))
            {
                nvalues.Add(item.Key, SystemTokenizer.GetNext());
            }
        }

        Trainer trainer = new Trainer(d.Item1, nvalues);
        trainer.TrainModel(epochs: 20);

        ChatBot bot = trainer.CreateNewBot();
        bot.Chat();
    }

    public static (TrainingData[], NGram_Model<string[]>, NGram_Model<string[]>) GetData()
    { 
        List<TrainingData> data = new List<TrainingData>();
        List<string> items = new List<string>();
        string path = Environment.CurrentDirectory + "/less_common_chats.txt";

        List<string> Inputs = new List<string>();
        List<string> Outputs = new List<string>();

        if (!string.IsNullOrEmpty(path))
        {
            string[] lines = File.ReadAllLines(path);
            if (lines != null)
            {
                for (int i = 0; i < lines.Length; i++)
                {
                    if (!string.IsNullOrEmpty(lines[i]))
                    {
                        items.Add(lines[i]);
                    }
                }

                for (int j = 0; j < items.Count; j += 2)
                {
                    var nt1 = items[j].Split(':');
                    var nt2 = items[j + 1].Split(":");
                    Inputs.Add(nt1[1].ToLower());
                    Outputs.Add(nt2[1].ToLower());
                }
            }
        }
        else
        {
            throw new Exception();
        }

        var filtered_inputs = NGram_Generator.Pad_Filter(Inputs.ToArray(), true, true);
        var filtered_outputs = NGram_Generator.Pad_Filter(Outputs.ToArray(), true, true);

        var input_ngrams = NGram_Generator.Generate(filtered_inputs, 2);
        var output_ngrams = NGram_Generator.Generate(filtered_outputs, 2);

        for (int i = 0; i < filtered_inputs.Length; i++)
        {
            data.Add(new TrainingData() 
            { 
                ExpectedOutput = filtered_outputs[i], 
                Input = filtered_inputs[i] 
            });
        }

        return (data.ToArray(), input_ngrams, output_ngrams);
    }
}
