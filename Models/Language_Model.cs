namespace TransformerTest.Models
{
    public abstract class Language_Model<T>
    {
        public T raw_text { get; set; }

        public string[] split_text { get; set; }

        public Dictionary<string, int> vocabulary_dict { get; set; }
    }
}
