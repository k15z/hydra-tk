using System;
using System.IO;

namespace HydraTK
{
   class SanityCheck
   {
      public static void Main() {
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
   }
}
