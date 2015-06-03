using System;
using System.IO;

/**
 * This file contains my implementation of the multi-layer perceptron, a type
 * of feedforward neural network which has been used for a wide variety of
 * artificial intelligence tasks.
 */
namespace HydraTK
{
   /**
    * The MultiLayerPerceptron is a feedforward neural network which uses the
    * sigmoid function as its activation function and supports a configurable
    * number of layers, as well as a configurable number of nodes in each of
    * the layers. It trains using backpropagation (steepest gradient descent)
    * and implements the NeuralNetwork interface.
    */
   class MultiPerceptron : NeuralNetwork
   {
      // Training Parameters
      public bool VERBOSE = true;
      public int MAX_ATTEMPT = 100000;
      public int RESCORE_INTERVAL = 1;
      public float LEARN_RATE = 0.01f;
      public float ERROR_MARGIN = 0.5f;
      public float PERCENT_CORRECT = 0.95f;

      // Network Parameters
      private int L;
      private int[] size;
      private float[][] n_node;
      private float[][] a_node;
      private float[][] bkprop;
      private float[][,] weight;

      /**
       * Create a new MultiLayerPerceptron object using the parameters found in
       * the `design` array. The `design` array contains the number of nodes in
       * each layer of the network; the length of the `design` array is the
       * number of nodes in the network.
       */
      public MultiPerceptron (int[] design)
      {
         L = design.Length;
         size = design;

         n_node = new float[L][];
         a_node = new float[L][];
         bkprop = new float[L][];
         for (int l = 0; l < L; l++) {
            n_node [l] = new float[size [l]];
            a_node [l] = new float[size [l]];
            bkprop [l] = new float[size [l]];
         }

         weight = new float[L - 1][,];
         for (int l = 0; l < L - 1; l++) {
            weight [l] = new float[size [l], size [l + 1]];
            for (int j = 0; j < size [l]; j++)
               for (int k = 0; k < size [l + 1]; k++)
                  weight [l] [j, k] = random ();
         }
      }

      /**
       * Feed data forwards through the network, storing intermediate values in
       * the `n_node` array to make backpropagation (training) faster. The
       * input vector is copied-by-reference directly into the 0th layer; the
       * other layers are calculated in place, and have the sigmoid function
       * applied to their output, which is stored in the `a_node` array.
       */
      public float[] feed (float[] input)
      {
         a_node [0] = input;
         for (int l = 1; l < L; l++)
            for (int k = 0; k < size [l]; k++) {
               n_node [l] [k] = 0.0f;
               for (int j = 0; j < size [l - 1]; j++)
                  n_node [l] [k] += a_node [l - 1] [j] * weight [l - 1] [j, k];
               a_node [l] [k] = sigmoid (n_node [l] [k]);
            }
         return a_node [L - 1];
      }

      /**
       * Trains the neural network on the input/output data sets until either
       * the score exceeds PERCENTAGE_CORRECT or the number of training cycles
       * exceeds MAX_ATTEMPTS. Each training cycle iterates over each training
       * model and adjusts each weight in the network using steepest gradient
       * descent (with backpropagation). After every RESCORE_INTERVAL of
       * training cycles, the score is recomputed and tested; if VERBOSE is set
       * to true, then the score is printed to the console.
       */
      public bool train (float[][] input, float[][] output)
      {
         int N = input.Length;
         int MAX_SCORE = N * size [L - 1];

         float error = 1.0f;
         int attempts = 0, score = 0;
         while ((double)score / (double)MAX_SCORE < PERCENT_CORRECT && attempts++ < MAX_ATTEMPT) {
            score = 0;
            error = 0.0f;
            for (int n = 0; n < N; n++) {
               var F = feed (input [n]);
               var T = output [n];

               for (int i = 0; i < size [L - 1]; i++)
                  bkprop [L - 1] [i] = (T [i] - F [i]) * dSigmoid (n_node [L - 1] [i]);

               for (int l = L - 2; l >= 0; l--) {
                  for (int j = 0; j < size [l]; j++) {
                     float value = 0.0f;
                     for (int i = 0; i < size [l + 1]; i++) {
                        value += bkprop [l + 1] [i] * weight [l] [j, i];
                        weight [l] [j, i] += LEARN_RATE * a_node [l] [j] * bkprop [l + 1] [i];
                     }
                     bkprop [l] [j] = value * dSigmoid (n_node [l] [j]);
                  }
               }

               if (attempts % RESCORE_INTERVAL == 0)
                  for (int k = 0; k < size [L - 1]; k++) {
                     if (Math.Abs (F [k] - T [k]) < ERROR_MARGIN)
                        score++;
                     error += Math.Abs (F [k] - T [k]);
                  }
            }

            if (VERBOSE && attempts % RESCORE_INTERVAL == 0) {
               Console.Write ("Attempt: " + attempts + "/" + MAX_ATTEMPT + ", ");
               Console.Write ("Score: " + score + "/" + MAX_SCORE + ", ");
               Console.Write ("Error: " + error);
               Console.WriteLine ();
            }
         }

         return ((double)score / (double)MAX_SCORE > PERCENT_CORRECT);
      }

      /**
       * Write the network design and network weights to the stream. These
       * values should be sufficient for restoring the network. The current
       * encoding is fully functional, but it is extremely inefficient.
       */
      public void toFile (StreamWriter o)
      {
         o.WriteLine (string.Join (" ", size));
         for (int layer = 0; layer < L - 1; layer++)
            for (int i = 0; i < size [layer]; i++)
               for (int j = 0; j < size [layer + 1]; j++)
                  o.Write (weight [layer] [i, j] + "\n");
         o.Close ();
      }

      /**
       * Restore a network from the stream. This method corresponds to the
       * `toFile` method, and should successfully restore the network design
       * and network weights from the file. It does not perform any error
       * checking; too few weights will result in a null pointer exception and
       * too many weights will be ignored.
       */
      public NeuralNetwork fromFile (StreamReader s)
      {
         int[] size = Array.ConvertAll (s.ReadLine ().Split (' '), int.Parse);
         MultiPerceptron n = new MultiPerceptron (size);
         for (int layer = 0; layer < n.L - 1; layer++)
            for (int i = 0; i < n.size [layer]; i++)
               for (int j = 0; j < n.size [layer + 1]; j++)
                  n.weight [layer] [i, j] = float.Parse (s.ReadLine ());
         s.Close ();
         return n;
      }

      /* Useful Little Self-Explanatory Methods */

      private Random rng = new Random ();

      private float random ()
      {
         return (float)(rng.NextDouble ()) * 2.0f - 1.0f;
      }

      private float sigmoid (float x)
      {
         return 1.0f / (1.0f + (float)Math.Exp (-x));
      }

      private float dSigmoid (float x)
      {
         x = sigmoid (x);
         return x * (1.0f - x);
      }

   }
}
