namespace NeuralNetwork
{
    //The idea is to initialize the weights in such a way that the variance remains the same
    //across different layers, which helps in training deep networks. 

    //https://www.pinecone.io/learn/weight-initialization/

    public class WeightInitializer
    {
        public enum DistributionType { Normal, Uniform }

        private readonly Random rand;

        public WeightInitializer(bool IsRepeatableTraining)
        {
            rand = IsRepeatableTraining ? new Random(1) : new Random();
        }

        public double GetWeight(int fanIn, int fanOut, bool isRelu, DistributionType distributionType)
        {
            if (distributionType == DistributionType.Uniform)
                return GetWeight_From_UniformDistribution(fanIn, fanOut, isRelu);
            if (distributionType == DistributionType.Normal)
                return GetWeight_From_NormalDistribution(fanIn, fanOut, isRelu);
            throw new NotImplementedException();

        }

        private double GetWeight_From_UniformDistribution(int fanIn, int fanOut, bool isRelu)
        {
            if (isRelu) //linear case for relu
            {
                //'He Kaiming' uniform initialization
                double limit = Math.Sqrt(12.0 / (fanIn + fanOut));
                return (UniformRandom() * 2 - 1) * limit;
            }
            else //nonlinear case for sigmoid, tanh
            {
                //'Xavier/Glorot' uniform initialization
                double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
                return (UniformRandom() * 2 - 1) * limit;
            }
        }

        private double GetWeight_From_NormalDistribution(int fanIn, int fanOut, bool isRelu)
        {
            if (isRelu) //linear case for relu
            {
                //'He Kaiming' normal initialization
                var stdDev = Math.Sqrt(4.0 / (fanIn + fanOut));
                return GaussianRandom() * stdDev;
            }
            else //nonlinear case for sigmoid, tanh
            {
                //'Xavier/Glorot' normal initialization
                double stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));
                return GaussianRandom() * stdDev;
            }
        }

        private double GaussianRandom()
        {
            double x1 = UniformRandom();
            double x2 = UniformRandom();
            return Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2); // Box-Muller transform
        }

        private double UniformRandom()
        {
            return 1 - rand.NextDouble();  // Uniform(0,1] random doubles (i.e. zero is excluded)
        }
    }
}
