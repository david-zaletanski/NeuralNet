using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNet
{
    class NeuralNetwork
    {
        bool Batch = false;
        double LearningRate = 0.7;
        double Momentum = 0.3;
        int MaxEpochs = 1000;

        int size;
        double[] Input;
        double[] Hidden;
        double[] Output;
        double[] OutputDelta;
        double[] HiddenDelta;
        double[,] wInputHidden;
        double[,] wHiddenOutput;
        double[,] wInputHiddenDelta;
        double[,] wHiddenOutputDelta;
        double[,] wInputHiddenGradient;
        double[,] wHiddenOutputGradient;

        double trainingSetAccuracy = 0.0;
        double trainingSetMSE = 0.0;
        double generalizationSetAccuracy = 0.0;
        double generalizationSetMSE = 0.0;

        public NeuralNetwork(int n)
        {
            size = n;
            Reset();
            RandomlyInitializeWeights(-0.5,0.5);
        }

        public void TrainNetwork(List<Tuple<double[], double[]>> TrainingCases, List<Tuple<double[], double[]>> GeneralizationCases, List<Tuple<double[], double[]>> ValidationCases, double desiredAccuracy)
        {
            int epoch = 0;
            while ((trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy) && epoch < MaxEpochs)
            {
                double prevTAcc = trainingSetAccuracy;
                double prevGAcc = generalizationSetAccuracy;

                RunTrainingEpoch(TrainingCases);

                generalizationSetAccuracy = GetSetAccuracy(GeneralizationCases);
                generalizationSetMSE = GetSetMSE(GeneralizationCases);

                // Only output significant changes.
                // if (Math.Ceiling(prevTAcc) != Math.Ceiling(trainingSetAccuracy) || Math.Ceiling(prevGAcc) != Math.Ceiling(generalizationSetAccuracy))
                Console.WriteLine("Epoch {0}: tA={1} gA={2} tMSE={3} gMSE={4}", epoch, trainingSetAccuracy, generalizationSetAccuracy, trainingSetMSE, generalizationSetMSE);

                epoch++;
            }

            double validationSetAccuracy = GetSetAccuracy(ValidationCases);
            double validationSetMSE = GetSetMSE(ValidationCases);
            Console.WriteLine("Validation Set Accuracy: {0}", validationSetAccuracy);
            Console.WriteLine("Validation Set MSE: {0}", validationSetMSE);
        }

        public void RunTrainingEpoch(List<Tuple<double[],double[]>> TrainingCases)
        {
            double incorrectPatterns = 0;
            double mse = 0;

            Batch = false;
            for (int i = 0; i < TrainingCases.Count; i++)
            {
                FeedForward(TrainingCases[i].Item1);
                //double MSE = GetMSE(Output, TrainingCases[i].Item2);
                BackPropagate(TrainingCases[i].Item2);

                bool patternCorrect = true;
                for (int k = 0; k < Output.Length; k++)
                {
                    if (GetRoundedOutputValue(Output[k]) != TrainingCases[i].Item2[k])
                        patternCorrect = false;

                    mse += Math.Pow(Output[k] - TrainingCases[i].Item2[k], 2);
                }
                if (!patternCorrect)
                    incorrectPatterns++;
                
                // Update Weights
                if(Batch)
                    UpdateWeights();

                trainingSetAccuracy = 100 - (incorrectPatterns / (double)TrainingCases.Count * 100.0);
                trainingSetMSE = mse / (Output.Length * TrainingCases.Count);
            }
        }

        public double[] FeedForward(double[] inputs)
        {
            if (Input.Length - 1 != inputs.Length)
                return null;

            // Input Layer
            for (int i = 0; i < inputs.Length; i++)
            {
                Input[i] = inputs[i];
            }
            // Hidden Layer
            for (int i = 0; i < Hidden.Length-1; i++)
            {
                Hidden[i] = 0.0;
                for (int j = 0; j < Input.Length; j++)
                {
                    Hidden[i] += Input[j] * wInputHidden[j, i];
                }
                Hidden[i] = Sigmoid(Hidden[i]);
            }
            // Output Layer
            for (int i = 0; i < Output.Length; i++)
            {
                Output[i] = 0.0;
                for (int j = 0; j < Hidden.Length; j++)
                {
                    Output[i] += Hidden[j] * wHiddenOutput[j, i];
                }
                Output[i] = Sigmoid(Output[i]);
            }

            return Output;
        }

        private void BackPropagate(double[] desired)
        {
            // Output Layer Node Delta
            for (int i = 0; i < Output.Length; i++)
            {
                OutputDelta[i] = GetOutputErrorGradient(desired[i], Output[i]);
                for (int j = 0; j < Hidden.Length; j++)
                {
                    if (!Batch)
                        wHiddenOutputDelta[j, i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * wHiddenOutputDelta[j, i];
                    else
                        wHiddenOutputDelta[j, i] += LearningRate * Hidden[j] * OutputDelta[i];
                }
            }

            // Hidden Layer Node Delta
            for (int i = 0; i < Hidden.Length-1; i++)
            {
                HiddenDelta[i] = GetHiddenErrorGradient(i);
                for (int j = 0; j < Input.Length; j++)
                {
                    if (!Batch)
                        wInputHiddenDelta[j, i] = LearningRate * Input[j] * HiddenDelta[i] + Momentum * wInputHiddenDelta[j, i];
                    else
                        wInputHiddenDelta[j, i] = LearningRate * Input[j] * HiddenDelta[i];
                }
            }

            if (!Batch)
                UpdateWeights();
        }

        private double GetOutputErrorGradient(double desired, double actual)
        {
            return actual * (1 - actual) * (desired - actual);
        }

        private double GetHiddenErrorGradient(int j)
        {
            double weightedSum = 0.0;
            for (int k = 0; k < Output.Length; k++)
                weightedSum += wHiddenOutput[j, k] * OutputDelta[k];
            return Hidden[j] * (1 - Hidden[j]) * weightedSum;
        }

        private double Sigmoid(double x)
        {
            return (1 / (double)(1 + Math.Pow(Math.E, -1 * x)));
        }
        private double SigmoidDerivative(double x)
        {
            double sig = Sigmoid(x);
            return sig * (1.0 - sig);
        }

        private void Reset()
        {
            Input = new double[size+1];
            for (int i = 0; i < Input.Length; i++)
                Input[i] = 0.0;
            Hidden = new double[size+1];
            for (int i = 0; i < Hidden.Length; i++)
                Hidden[i] = 0.0;
            Output = new double[size];
            for (int i = 0; i < Output.Length; i++)
                Output[i] = 0.0;
            wInputHidden = new double[size + 1, size];
            for (int i = 0; i < (size + 1); i++)
                for (int j = 0; j < size; j++)
                    wInputHidden[i, j] = 0.0;
            wHiddenOutput = new double[size + 1, size];
            for (int i = 0; i < (size + 1); i++)
                for (int j = 0; j < size; j++)
                    wHiddenOutput[i, j] = 0.0;
            OutputDelta = new double[size];
            for (int i = 0; i < size; i++)
                OutputDelta[i] = 0.0;
            HiddenDelta = new double[size+1];
            for (int i = 0; i < size+1; i++)
                HiddenDelta[i] = 0.0;
            wInputHiddenGradient = new double[size + 1, size];
            for (int i = 0; i < (size + 1); i++)
                for (int j = 0; j < size; j++)
                    wInputHiddenGradient[i, j] = 0.0;
            wHiddenOutputGradient = new double[size + 1, size];
            for (int i = 0; i < (size + 1); i++)
                for (int j = 0; j < size; j++)
                    wHiddenOutputGradient[i, j] = 0.0;
            wInputHiddenDelta = new double[size + 1, size + 1];
            for (int i = 0; i < (size + 1); i++)
                for (int j = 0; j < (size + 1); j++)
                    wInputHiddenDelta[i, j] = 0.0;
            wHiddenOutputDelta = new double[size + 1, size + 1];
            for (int i = 0; i < (size + 1); i++)
                for (int j = 0; j < (size + 1); j++)
                    wHiddenOutputDelta[i, j] = 0.0;


            // Bias
            Input[size] = 1.0;
            Hidden[size] = 1.0;
        }

        private void RandomlyInitializeWeights(double min, double max)
        {
            Random rnd = new Random();
            for (int i = 0; i < (size + 1); i++)
            {
                for (int j = 0; j < size; j++)
                {
                    wInputHidden[i, j] = rnd.NextDouble()*(max-min) + min;
                    wHiddenOutput[i, j] = rnd.NextDouble()*(max-min) + min;
                    wInputHiddenDelta[i, j] = 0.0;
                    wHiddenOutputDelta[i, j] = 0.0;
                }
            }
        }

        private void UpdateWeights()
        {
            for (int i = 0; i < Input.Length; i++)
            {
                for (int j = 0; j < Hidden.Length - 1; j++)
                {
                    wInputHidden[i, j] += wInputHiddenDelta[i, j];

                    if (Batch)
                        wInputHiddenDelta[i, j] = 0;
                }
            }
            for (int i = 0; i < Hidden.Length; i++)
            {
                for (int j = 0; j < Output.Length; j++)
                {
                    wHiddenOutput[i, j] += wHiddenOutputDelta[i, j];
                    if (Batch)
                        wHiddenOutputDelta[i, j] = 0;
                }
            }
        }

        private double GetRoundedOutputValue(double x)
        {
            if (x < 0.1) return 0.0;
            else if (x > 0.9) return 1.0;
            else return -1.0;
        }

        private double GetSetMSE(List<Tuple<double[], double[]>> set)
        {
            double mse = 0;
            for (int i = 0; i < set.Count; i++)
            {
                FeedForward(set[i].Item1);
                for (int k = 0; k < Output.Length; k++)
                {
                    mse += Math.Pow(Output[k] - set[i].Item2[k], 2);
                }
            }
            return mse / (Output.Length * set.Count);
        }

        private double GetSetAccuracy(List<Tuple<double[], double[]>> set)
        {
            double incorrectResults = 0;
            for (int i = 0; i < set.Count; i++)
            {
                FeedForward(set[i].Item1);

                bool correctResult = true;
                for (int k = 0; k < Output.Length; k++)
                {
                    if (GetRoundedOutputValue(Output[k]) != set[i].Item2[k])
                        correctResult = false;
                }
                if (!correctResult)
                    incorrectResults++;
            }
            return 100 - (incorrectResults / set.Count * 100);
        }
    }
}
