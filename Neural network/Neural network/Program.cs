namespace NeuralNetwork
{
	public class Program
	{
		public static void Main(string[] args)
		{
            //HEADER TEXT.
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($@"
______   ___   _____                            (
| ___ \ / _ \ |_   _|                       (   )  )
| |_/ // /_\ \  | |                          )  ( )
|  __/ |  _  |  | |                          .....
| |    | | | | _| |_                      .:::::::::.
\_|    \_| |_/ \___/                      ~\_______/~  (Yummy Pie)

PLAIN-ARTIFICIAL-INTELLIGENCE. CREATED BY ARE OLSEN, 01.08.2023.
-------------------------------------");

            //INIT NETWORK.
            Network? network = new Network(
                layerLengths: new int[] { 784, 100, 10 },
                isRepeatableTraining: true);

            //LOAD DATA FROM MNIST INTO ARRAYS.
            double[][] trainInput;
			double[][] trainOutput;

            double[][] testInput;
            double[][] testOutput;
            DataSetLoader.LoadCsvToArray("../../../mnist_train.csv", 1, out trainOutput, out trainInput, 255d);
            DataSetLoader.LoadCsvToArray("../../../mnist_test.csv", 1, out testOutput, out testInput, 255d);

            //FIX OUTPUT ARRAY, (DATASET DOESN'T USE ARRAY FOR OUTPUT, BUT 1 NUMBER :| ).
            for (int i = 0; i < trainOutput.Length; i++)
			{
				double[] temp = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				temp[Convert.ToInt32(trainOutput[i][0])] = 1;
				trainOutput[i] = temp;
			}

			for (int i = 0; i < testOutput.Length; i++)
			{
				double[] temp = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				temp[Convert.ToInt32(testOutput[i][0])] = 1d;
				testOutput[i] = temp;
			}

            //TRAIN NETWORK.
            Console.WriteLine("TRAINING STARTED."); 
			
            int epochNum = 0;
            
            network.Train(1,   15, 0.01, 0.95, 0, trainInput, trainOutput, testInput, testOutput, ref epochNum);
            network.Train(1,   18, 0.01, 0.90, 0, trainInput, trainOutput, testInput, testOutput, ref epochNum);
            network.Train(1,   24, 0.01, 0.90, 0, trainInput, trainOutput, testInput, testOutput, ref epochNum);
            network.Train(100, 32, 0.01, 0.90, 0, trainInput, trainOutput, testInput, testOutput, ref epochNum);

            //SAVE NETWORK.
            Serializer.SerializerBinary.SaveObjectToFile("../../../networkserialized", network);
            //network = Serializer.SerializerBinary.LoadObjectFromFile<Network>("../../../networkserialized");
            //network.Test(testInput, testOutput);
            //network.Test(testInput, testOutput);
        }
    }
}

