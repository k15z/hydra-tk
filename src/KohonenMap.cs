using System;
using System.IO;

/**
 * This file contains my implementation of the two-dimensional kohonen map, a
 * type of neural network which produces a 2d map of the input vectors while
 * preserving the topology of the inputs.
 */
namespace HydraTK
{
   /**
    * The Kohonen Self-Organizing Map is a neural network which performs
    * unsupervised learning by mapping input vectors into a 2d map; the SOM
    * preserves topology, meaning similar input vectors should end up close to
    * each other on the map.
    */
   class KohonenMap : Unsupervised
   {
      // Training Parameters
      public int NEIGHBOR_RADIUS;
      public float INITIAL_ALPHA = 1.0f;
      public int TOTAL_ITERATIONS = 1000;

      // Network Parameters
      private int W, H, L;
      private float[,][] weight;
      private Random rng = new Random ();

      /**
       * Create a new KohonenMap object of width W, height H, and input vector
       * size L. The values in each vector are initially set to a random number
       * between -1.0f and 1.0f.
       */
      public KohonenMap(int W, int H, int L)
      {
         NEIGHBOR_RADIUS = Math.Max(W, H);
         weight = new float[W,H][];
         this.W = W; this.H = H; this.L = L;
         for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++) {
               weight[x,y] = new float[L];
               for (int z = 0; z < L; z++)
                  weight[x,y][z] = (float)(rng.NextDouble ()) * 2.0f - 1.0f;
            }
      }

      /**
       * Seed the weights in the map with random values selected from the input
       * vectors. While this is useless for simple tasks such as the XOR
       * problem, it can be very helpful for things like optical character
       * recognition where many of the dimensions in the input vector are more
       * or less useless. (In optical character recognition, most of the values
       * will be white in all of the images; there is no point initializing the
       * kmap to black in a location where it will never happen.)
       */
      public void seed (float[][] input)
      {
         for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++)
               for (int z = 0; z < L; z++)
                  weight[x,y][z] = input[rng.Next(0,input.Length)][z];
      }

      /**
       * Iterate over all the vectors in the 2d weight map, calculating the
       * difference between the vector in the map and the input vector. The
       * location in the map which produces the smallest variance is returned
       * as a float array.
       */
      public float[] feed (float[] input)
      {
         float min_variance = float.MaxValue;
         int[] result = new int[2];
         for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++) {
               float variance = 0.0f;
               for (int z = 0; z < L; z++) {
                  float error = input[z] - weight[x,y][z];
                  variance += error * error;
               }
               if (variance < min_variance) {
                  min_variance = variance;
                  result[0] = x; result[1] = y;
               }
            }
         return new float[] {result[0], result[1]};
      }

      /**
       * Adjusts the map weights by iterating over the training data, finding
       * the best match for each input vector, and adjusting the best match as
       * well as its neighbors to bring them even closer to the input vector.
       */
      public void train (float[][] data)
      {
         int radius = NEIGHBOR_RADIUS;
         float alpha = INITIAL_ALPHA;
         float delta = INITIAL_ALPHA/TOTAL_ITERATIONS;
         while (alpha > 0) {
   			foreach (float[] input in data) {
               float[] result = feed(input);
   				int x = (int)result[0], y = (int)result[1];
   				for (int z = 0; z < L; z++) {
   					for (int _x = -radius; _x <= radius; _x++)
   						for (int _y = -radius; _y <= radius; _y++)
   							if (x+_x >= 0 && y+_y >= 0 && x+_x < W && y+_y < H)
   								weight[x+_x,y+_y][z] += (alpha * (input[z] - weight[x+_x,y+_y][z])) / (_x*_x + _y*_y + 1);
   				}
   			}

            radius = (int)(radius * alpha / INITIAL_ALPHA);
            alpha -= delta;
         }
      }

      /**
       * Write the network dimensions and network weights to the stream. These
       * values should be sufficient for restoring the network. The current
       * encoding is fully functional, but it is extremely inefficient.
       */
      public void toFile (StreamWriter o)
      {
         o.WriteLine (W + " " + H + " " + L);
         for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++)
               for (int z = 0; z < L; z++)
                  o.Write (weight [x,y] [z] + "\n");
         o.Close ();
      }

      /**
       * Restore a network from the stream. This method corresponds to the
       * `toFile` method, and should successfully restore the network
       * dimensions and network weights from the file. It does not perform any
       * error checking; too few weights will result in a null pointer
       * exception and too many weights will be ignored.
       */
      public HydraNode fromFile (StreamReader s)
      {
         int[] size = Array.ConvertAll (s.ReadLine ().Split (' '), int.Parse);
         KohonenMap kmap = new KohonenMap(size[0], size[1], size[2]);
         for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++)
               for (int z = 0; z < L; z++)
                  weight [x,y] [z] = float.Parse (s.ReadLine ());
         return kmap;
      }
   }
}
