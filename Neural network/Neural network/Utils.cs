namespace NeuralNetwork
{
    internal class Utils
    {
        public const double Epsilon = 1.0 / int.MaxValue; // -> the loss in bits is max 31 bits

        public static void ThrowWhenBadValue(double value)
        {
            if (!double.IsNormal(value) && value != 0.0)
                throw new Exception();
        }
    }
}
