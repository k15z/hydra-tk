using System;
using System.IO;

/**
 * This file contains the various interfaces which will be implemented by the
 * various objects included in the HydraTK namespace. As of 06/02/2015, the
 * only interface that is being used is the NeuralNetwork interface.
 */
namespace HydraTK
{
    /**
     * Objects which implement the NeuralNetwork interface must accept an array
     * of floating point numbers as the input vector and return an array of
     * floating point numbers as the output vector.
     */
    interface NeuralNetwork
    {
        float[] feed(float[] input);
        void toFile(StreamWriter stream);
        NeuralNetwork fromFile(StreamReader stream);
    }

    /**
     * TBD.
     */
    interface ClassificationTree
    {
    }

    /**
     * TBD.
     */
    interface RegressionTree
    {
    }
}
