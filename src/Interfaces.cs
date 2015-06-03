using System;
using System.IO;

/**
* This file contains the various interfaces which will be implemented by the
* various objects included in the HydraTK namespace.
*/
namespace HydraTK
{
   interface HydraNode
   {
      float[] feed(float[] input);
   }

   interface Supervised : HydraNode
   {
      void train(float[][] input, float[][] output);
   }

   interface Unsupervised : HydraNode
   {
      void train(float[][] input);
   }
}
