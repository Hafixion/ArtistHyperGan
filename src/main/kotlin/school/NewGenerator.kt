package school

import batchLayer
import convLayer
import deconvLayer
import dropOutLayer
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions
import outputLayer
import poolingLayer
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.roundToInt

val genLayers = arrayOf(
    deconvLayer(
        32, 5, 2, 0, Activation.SELU, convolutionMode = ConvolutionMode.Same,
        channels = 64
    ),
    batchLayer(),

    deconvLayer(32, 5, 2, 0, Activation.SELU, convolutionMode = ConvolutionMode.Same),
    batchLayer(),

    deconvLayer(16, 5, 2, 0, Activation.SELU, convolutionMode = ConvolutionMode.Same),
    batchLayer(),

    deconvLayer(8, 5, 2, 0, Activation.SELU, convolutionMode = ConvolutionMode.Same),
    batchLayer(),
    deconvLayer(3, 5, 1, 0, Activation.TANH, convolutionMode = ConvolutionMode.Same)
)

val disLayers = arrayOf(
    convLayer(8, 5, 2, activation = Activation.RELU, channels = 3, convolutionMode = ConvolutionMode.Same),

    dropOutLayer(0.4, updater=null),

    convLayer(16, 5, 2, activation = Activation.RELU, convolutionMode = ConvolutionMode.Same),

    dropOutLayer(0.4, updater=null),

    convLayer(32, 5, 2, activation = Activation.RELU, convolutionMode = ConvolutionMode.Same),

    dropOutLayer(0.4, updater=null),

    poolingLayer(SubsamplingLayer.PoolingType.MAX),

    outputLayer(1, Activation.SIGMOID, LossFunctions.LossFunction.XENT, null),
)

val frozenDisLayers = arrayOf(
    convLayer(8, 5, 2, activation = Activation.RELU, channels = 3, convolutionMode = ConvolutionMode.Same, updater = Sgd.builder().learningRate(0.0).build()),

    dropOutLayer(0.4, updater=null),

    convLayer(16, 5, 2, activation = Activation.RELU, convolutionMode = ConvolutionMode.Same, updater = Sgd.builder().learningRate(0.0).build()),

    dropOutLayer(0.4, updater=null),

    convLayer(32, 5, 2, activation = Activation.RELU, convolutionMode = ConvolutionMode.Same, updater = Sgd.builder().learningRate(0.0).build()),

    dropOutLayer(0.4, updater=null),

    convLayer(64, 5, 2, activation = Activation.RELU, convolutionMode = ConvolutionMode.Same, updater = Sgd.builder().learningRate(0.0).build()),

    dropOutLayer(0.4, updater=null),

    outputLayer(1, Activation.SIGMOID, LossFunctions.LossFunction.XENT, updater = Sgd.builder().learningRate(0.0).build()),
)

fun getGenerator(): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder().apply {
        seed(123)
        updater(Adam(0.002))
        weightInit(WeightInit.XAVIER)
        activation(Activation.IDENTITY)
    }.list().apply {
        inputType = InputType.convolutional(7, 7, 64)

        genLayers.forEach { layer(it) }
    }.build()
}

fun getDiscriminator(): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder().apply {
        seed(123)
        updater(Adam(0.002))
        weightInit(WeightInit.XAVIER)
        activation(Activation.IDENTITY)
    }.list().apply {
        inputType = InputType.convolutional(112, 112, 3)

        disLayers.forEach { layer(it) }
    }.build()
}

fun getGan(): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder().apply {
        seed(123)
        updater(Adam(0.002))
        weightInit(WeightInit.XAVIER)
        activation(Activation.IDENTITY)
    }.list().apply {
        inputType = InputType.convolutional(7, 7, 64)

        genLayers.forEach { layer(it) }
        frozenDisLayers.forEach { layer(it) }
    }.build()
}

fun copyParamsToGen(gen: MultiLayerNetwork, gan: MultiLayerNetwork) {
    gen.layers.forEachIndexed { index, layer ->
        layer.setParams(gan.getLayer(index).params())
    }
}

fun updateGanFromDis(dis: MultiLayerNetwork, gan: MultiLayerNetwork) {
    gan.layers.forEachIndexed { index, layer ->
        if (index >= genLayers.size) layer.setParams(dis.getLayer(index - genLayers.size).params())
    }
}

fun generateRGBImage(size: Int, red: Double = 1.0, green: Double = 0.0, blue: Double = 0.0): INDArray {
    val tensor = Nd4j.create(3, size, size)

    for (i in 0 until size) for (j in 0 until size) {
        tensor.putScalar(intArrayOf(0, i, j), red)
        tensor.putScalar(intArrayOf(1, i, j), green)
        tensor.putScalar(intArrayOf(2, i, j), blue)
    }

    return tensor
}

fun saveImageFromTensor(tensor: INDArray, name: String) {
    val size = tensor.shape()[1].toInt()
    val img = BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)

    for (i in 0 until size) for (j in 0 until size) {

        val red = (tensor.getDouble(0, i, j) * 255).roundToInt()
        val green = (tensor.getDouble(1, i, j) * 255).roundToInt()
        val blue = (tensor.getDouble(2, i, j) * 255).roundToInt()

        val color = Color(red, green, blue).rgb
        img.setRGB(i, j, color)
    }

    ImageIO.write(img, "jpg", File(name))
}

fun getBatchDataSetFromTensors(list: List<INDArray>): INDArray {
    val shape = list.first().shape()

    val height = shape[1].toInt()
    val width = shape[2].toInt()
    val depth = shape[0].toInt()

    val batchSize = list.size
    val tensor = Nd4j.zeros(batchSize, depth, height, width)

    for ((index, array) in list.withIndex()) for (i in 0 until height) for (j in 0 until width) for (z in 0 until depth) {
        tensor.putScalar(intArrayOf(index, z, i, j), array.getDouble(z, i, j))
    }

    return tensor
}

/**
 * copy parameters from GAN to Generator and Discriminator
 * @param gen
 * @param dis
 * @param gan
 */
fun copyParamsToGan(gen: MultiLayerNetwork, dis: MultiLayerNetwork, gan: MultiLayerNetwork) {
    val genLayerCount = gen.layers.size
    for (i in gan.layers.indices) {
        if (i < genLayerCount) {
            gen.getLayer(i).setParams(gan.getLayer(i).params())
        } else {
            // this will not affect anything, since gan-dis parameters are shallow copy and originated from dis
            dis.getLayer(i - genLayerCount).setParams(gan.getLayer(i).params())
        }
    }
}

/**
 * update the discriminator parameters in GAN
 * @param gen
 * @param dis
 * @param gan
 */
fun updateGanFromDis(gen: MultiLayerNetwork, dis: MultiLayerNetwork, gan: MultiLayerNetwork) {
    val genLayerCount = gen.layers.size
    for (i in genLayerCount until gan.layers.size) {
        gan.getLayer(i).setParams(dis.getLayer(i - genLayerCount).params())
    }
}