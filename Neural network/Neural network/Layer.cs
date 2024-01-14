using System.Diagnostics;
using System.Reflection.Emit;

namespace NeuralNetwork
{
    //Layer of nodes.
    [Serializable]
    public class Layer
    {
        public Node[] nodes;
        public bool outputLayer;
        public bool inputLayer;
        public Layer(Node[] nodes, bool outputLayer, bool inputLayer)
        {
            this.nodes = nodes;
            this.outputLayer = outputLayer;
            this.inputLayer = inputLayer;
        }

        /*
            UPDATE LAYER DATA.
        */
        public void UpdateValues()
        {
            if (outputLayer)
                UpdateValues_On_OutputLayer();
            else
                UpdateValues_On_HiddenLayer();
        }

        private void UpdateValues_On_HiddenLayer()
        {
            for (int i = 0; i < nodes.Length; i++)
            {
                var node = nodes[i];
                var netInput = node.GetNetActivationInput();
                node.value = Activations.LeakyRELU.Activation(netInput);
                Utils.ThrowWhenBadValue(node.value);
            }
        }

        private void UpdateValues_On_OutputLayer()
        {
            double[] netInputs = this.GetLayerNetInputs();
            for (int i = 0; i < nodes.Length; i++)
            {
                var node = nodes[i];
                node.value = Activations.SoftMax.Activation(netInputs, netInputs[i]);
                Utils.ThrowWhenBadValue(node.value);
            }
        }

        public void UpdateLayerGradients(double[] expected)
        {
            for (int j = 0; j < nodes.Length; j++)
            {
                Node node = nodes[j];
                if (outputLayer)
                    node.UpdateGradient(this, expected[j]);
                else
                    node.UpdateGradient();
                node.AddBiasDerivative();
                foreach (Connection connection in node.inputConnections)
                {
                    connection.AddWeightDerivative();
                }
            }
        }

        /*
            SET LAYER DATA.
        */
        public void SetLayerValues(double[] values)
        {
            Debug.Assert(inputLayer);
            for (int i = 0; i < values.Length; i++)
            {
                nodes[i].value = values[i];
            }
        }

        /*
            GET LAYER DATA.
        */
        public double[] GetLayerValues()
        {
            double[] vals = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                vals[i] = nodes[i].value;
            }
            return vals;
        }

        public double[] GetLayerNetInputs()
        {
            double[] nets = new double[nodes.Length];
            for(int i = 0;  i < nets.Length; i++)
            {
                nets[i] = nodes[i].GetNetActivationInput();
            }
            return nets;
        }
    }
}

