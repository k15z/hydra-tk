using System;
using System.IO;

namespace HydraTK
{
   class SanityCheck
   {
      public static void Main() {
         
      }

      static void MLP() {
         MultiPerceptron mlp = new MultiPerceptron(new int[]{2,7,1});
         float[][] input = new float[][] {
            new float[] {0.0f, 0.0f},
            new float[] {0.0f, 1.0f},
            new float[] {1.0f, 0.0f},
            new float[] {1.0f, 1.0f}
         };
         float[][] output = new float[][] {
            new float[] {0.0f},
            new float[] {1.0f},
            new float[] {1.0f},
            new float[] {0.0f}
         };
         mlp.train(input, output);
      }

      static void KMAP() {
         KohonenMap kmap = new KohonenMap(5, 5, 2);
         float[][] input = new float[][] {
            new float[] {0.0f, 0.0f},
            new float[] {0.0f, 1.0f},
            new float[] {1.0f, 0.0f},
            new float[] {1.0f, 1.0f}
         };
         kmap.train(input);

         Random rng = new Random();
         foreach(float[] row in input) {
            Console.WriteLine(string.Join(" ", row) + " => " + string.Join(" ", kmap.feed(row)));
            for (int i = 0; i < 3; i++) {
               row[0] = (float)(row[0] + rng.NextDouble()/2 - 0.5f);
               row[1] = (float)(row[1] + rng.NextDouble()/2 - 0.5f);
               Console.WriteLine("\t" + string.Join(" ", row) + " => " + string.Join(" ", kmap.feed(row)));
            }
         }
      }
   }
}
