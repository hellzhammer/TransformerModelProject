namespace TransformerTest.Models
{
    public class NGram_Time_Series
    {
        public int n_gram_count { get; set; }

        public Dictionary<double[], int> n_grams { get; set; }
        public Dictionary<double[], int> n_minus_one_gram { get; set; }
        public Dictionary<double, int> token_dict { get; set; }
    }
}
