namespace TransformerTest.Models
{
    public class NGram_Model<T> : Language_Model<T>
    {
        public int n_gram_count { get; set; }

        public Dictionary<string, int> n_grams { get; set; }
    }
}
