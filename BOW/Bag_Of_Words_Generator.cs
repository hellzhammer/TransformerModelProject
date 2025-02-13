using TransformerTest.Models;

public class Bag_Of_Words_Generator
{
    public static BOW_Model<string[]> Generate(string[] raw_input)
    {
        BOW_Model<string[]> model = new BOW_Model<string[]>();
        model.raw_text = raw_input;
        model.vocabulary_dict = new Dictionary<string, int>();

        for (int i = 0; i < raw_input.Length; i++)
        {
            var split = raw_input[i].Split(" ");
            for (int y = 0; y < split.Length; y++)
            {
                if (!model.vocabulary_dict.ContainsKey(split[y])
                    && !string.IsNullOrWhiteSpace(split[y]))
                {
                    model.vocabulary_dict.Add(split[y], 1);
                }
                else if (model.vocabulary_dict.ContainsKey(split[y])
                    && !string.IsNullOrWhiteSpace(split[y]))
                {
                    model.vocabulary_dict[split[y]]++;
                }
            }
        }

        return model;
    }
}