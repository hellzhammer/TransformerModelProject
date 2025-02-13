public static class Dataset
{
    private static TrainingData[] GetData()
    {
        List<TrainingData> data = new List<TrainingData>();
        List<string> items = new List<string>();
        string path = Environment.CurrentDirectory + "/less_common_chats.txt";

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

                for (int j = 0; j < items.Count; j+=2)
                {
                    var nt1 = items[j].Split(':');
                    var nt2 = items[j+1].Split(":");
                    var tr = new TrainingData { Input = nt1[1], ExpectedOutput = nt2[1] };
                    data.Add(tr);
                }
            }
        }
        else
        {
            throw new Exception();
        }

        return data.ToArray();
    }
}