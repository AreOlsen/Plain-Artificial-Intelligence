using System;
using System.Globalization;

namespace NeuralNetwork
{
    public class DataSetLoader
    {
        //FILE, A=LENGTH OF START VALUES THAT IS EXPECTED OUTPUT, arrayA= EXPECTED OUTPUT ARRAY, arrayB= INPUT ARRAY,  nodeDividerOutput=INPUT VALUE DIVIDER.
        public static void LoadCsvToArray(string filePath, int A, out double[][] arrayA, out double[][] arrayB, double nodeDividerOutput = 1d)
        {
            List<double[]> tempArrayA = new List<double[]>();
            List<double[]> tempArrayB = new List<double[]>();

            using (var reader = new StreamReader(filePath))
            {
                string? line;
                int lineCount = 0;

                while ((line = reader.ReadLine()) != null)
                {
                    if (lineCount == 0)
                    {
                        lineCount++;
                        continue; // Skip the first row, usually just labels.
                    }

                    string[] values = line.Split(',');

                    if (values.Length < A)
                    {
                        Console.WriteLine($"Warning: Line {lineCount + 1} contains less than {A} elements.");
                        continue;
                    }

                    double[] rowArrayA = new double[A];
                    double[] rowArrayB = new double[values.Length - A];

                    for (int i = 0; i < values.Length; i++)
                    {
                        double nodeValue;

                        if (!double.TryParse(values[i], NumberStyles.Float, CultureInfo.InvariantCulture, out nodeValue))
                        {
                            Console.WriteLine($"Error: Line {lineCount}, Element {i} is not a valid double value.");
                            continue;
                        }

                        if (i < A)
                            rowArrayA[i] = nodeValue;
                        else
                            rowArrayB[i - A] = nodeValue / nodeDividerOutput;
                    }

                    tempArrayA.Add(rowArrayA);
                    tempArrayB.Add(rowArrayB);

                    lineCount++;
                }
            }

            arrayA = tempArrayA.ToArray();
            arrayB = tempArrayB.ToArray();
        }

        /*
        Shuffling the samples before training a neural network is a common practice in machine learning,
        including deep learning, and serves several important purposes:

        1. Avoiding Order Bias: If the training data is ordered in a certain way, and you feed the data 
           to the network without shuffling, the model may inadvertently learn patterns based on the 
           order of the samples rather than the inherent patterns in the data. Shuffling helps to break 
           any correlation between the order of the data and the learning process, preventing the model 
           from being influenced by the order of presentation.

        2. Improving Generalization: Shuffling ensures that each mini-batch used during training is a 
           representative sample of the overall dataset. This helps the model generalize better to unseen 
           data because it encounters a diverse set of examples in each batch, rather than learning specific 
           patterns from a contiguous block of similar samples.

        3. Enhancing Learning Dynamics: Shuffling the data can improve the learning dynamics of the neural 
           network. In stochastic gradient descent (SGD) or variants like mini-batch gradient descent, the 
           model updates its weights based on small batches of data. Shuffling helps in creating more diverse 
           and representative batches, which can lead to a more stable convergence during training.

        4. Preventing Model Overfitting: Without shuffling, the model might overfit to specific patterns 
           present in a fixed order of samples. Shuffling introduces randomness and variability, which helps 
           prevent the model from memorizing the order-specific details and encourages it to learn more 
           general and robust features.
        */

        private static Random rnd = new Random(0);
        public static void Shuffle(double[][] arrayA, double[][] arrayB)
        {
            for (int i = 0; i < arrayA.Length; i++)
            {
                var index = rnd.Next(0, arrayA.Length);
                var currentA = arrayA[i];
                var currentB = arrayB[i];
                arrayA[i] = arrayA[index];
                arrayB[i] = arrayB[index];
                arrayA[index] = currentA;
                arrayB[index] = currentB;
            }
            Console.WriteLine("Shuffled");
        }

    }
}

