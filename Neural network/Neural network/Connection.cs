namespace NeuralNetwork
{
    //Connection from one node to another.
    [Serializable]
    public class Connection
	{
		public Node nodeIn;
		public Node nodeOut;
		public double weight;
		public double weightDerivative = 0d;
		public double weightVelocity = 0d;

		public Connection(Node nodeIn, Node nodeOut, double weight = 1d)
		{
			this.nodeIn = nodeIn;
			this.nodeOut = nodeOut;
			this.weight = weight;
		}

		/*
			UPDATE CONNECTION DATA.
		*/
		public void AddWeightDerivative()
		{

			weightDerivative += nodeOut.gradient * nodeIn.value;
        }

		public void UpdateWeight(double trainingStep, double momentum = 0.9d, double regularization = 0.1d)
		{
			double weightDecay = (1 - regularization * trainingStep); 
			double velocity = weightVelocity * momentum - weightDerivative * trainingStep;
			weightVelocity = velocity;
			weight = weight * weightDecay + velocity;
			Utils.ThrowWhenBadValue(weight);
            weightDerivative = 0d;
		}
	}
}

