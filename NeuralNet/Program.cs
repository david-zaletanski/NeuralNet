using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork NN = new NeuralNetwork(2);
            List<Tuple<double[], double[]>> TrainingData = GenerateXORTestData(500);
            List<Tuple<double[], double[]>> GeneralizationData = GenerateXORTestData(50);
            List<Tuple<double[], double[]>> ValidationData = GenerateXORTestData(10);
            NN.TrainNetwork(TrainingData, GeneralizationData, ValidationData, 99.0);

            Console.ReadLine();
        }

        static List<Tuple<double[], double[]>> GenerateXORTestData(int n)
        {
            List<Tuple<double[], double[]>> Result = new List<Tuple<double[], double[]>>();

            // Add Basic XOR Cases
            Result.Add(new Tuple<double[], double[]>(new double[] { 0, 0 }, new double[] { 0, 0 }));
            Result.Add(new Tuple<double[], double[]>(new double[] { 0, 1 }, new double[] { 1, 0 }));
            Result.Add(new Tuple<double[], double[]>(new double[] { 1, 0 }, new double[] { 1, 0 }));
            Result.Add(new Tuple<double[], double[]>(new double[] { 1, 1 }, new double[] { 0, 0 }));

            // Generate extra cases (randomly from base 4)
            Random rnd = new Random();
            for (int i = 0; i < n; i++)
            {
                Result.Add(Result[rnd.Next(4)]);
            }

            return Result;
        }
    }
}
