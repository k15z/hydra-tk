using System;
using System.IO;

/**
 * This file contains the various interfaces which will be implemented by the
 * various objects included in the HydraTK namespace. As of 06/02/2015, the
 * only interface that is being used is the Node interface.
 */
namespace HydraTK
{
    /**
     * Objects which implement the HydraNode interface must accept an array
     * of floating point numbers as the input vector and return an array of
     * floating point numbers as the output vector.
     */
    interface HydraNode
    {
        float[] feed(float[] input);
    }

    /**
     * Objects which implement the Regressor interface must accept an array
     * of floating point numbers as the input vector and return a single
     * floating point number as the output. The Regressor interface inherits
     * from HydraNode.
     */
    interface Regressor : HydraNode
    {
        float regress(float[] input);
    }

    /**
     * Objects which implement the Classifier interface must accept an array
     * of floating point numbers as the input vector and return a single
     * integer indicating the category. The Classifier interface inherits
     * from HydraNode.
     */
    interface Classifier : HydraNode
    {
        int classify(float[] input);
    }
}
