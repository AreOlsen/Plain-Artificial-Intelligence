namespace NeuralNetwork
{
    public class WeightInitializer
    {
        private readonly Random rand;

        public WeightInitializer(bool IsRepeatableTraining)
        {
            rand = IsRepeatableTraining ? new Random(1) : new Random();
        }

        public double GetWeight(int prevLayerLength)
        {
            double weight = RandomInNormalDistribution(0, 1) / Math.Sqrt(prevLayerLength);
            return weight;

        }

        private double RandomInNormalDistribution(double mean, double standardDeviation)
        {
            double x1 = 1 - rand.NextDouble();
            double x2 = 1 - rand.NextDouble();

            double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * standardDeviation + mean;
        }
    }
}
