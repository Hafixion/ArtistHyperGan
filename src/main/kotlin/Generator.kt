
import org.apache.commons.lang.ArrayUtils
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation.*
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.IUpdater
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.XENT

fun getCombinedNetwork(seed: Long = 123): MultiLayerConfiguration {
    val gen = getGeneratorLayers()
    val dis = getDiscriminatorLayers()
    val layers = ArrayUtils.addAll(gen, dis) as Array<Layer>

    return NeuralNetConfiguration.Builder().apply {
        seed(seed)
        updater(Adam.Builder().learningRate(0.0002).beta1(0.5).build())
        weightInit(WeightInit.XAVIER)
        activation(IDENTITY)
        gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
        gradientNormalizationThreshold(100.0)
    }.list().apply {
        for (layer in layers) layer(layer)

        InputType.convolutional(7, 7, 64)
    }.build()
}

fun getGenerator(seed: Long = 123, updater: IUpdater = Adam.Builder().learningRate(0.0002).beta1(0.5).build()): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder().apply {
        seed(seed)
        updater(updater)
        gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
        gradientNormalizationThreshold(100.0)
        weightInit(WeightInit.XAVIER)
        activation(IDENTITY)
    }.list().apply {
        for (layer in getGeneratorLayers(seed)) layer(layer)

        InputType.convolutional(7, 7, 64)
    }.build()
}

fun getDiscriminator(seed: Long = 123, updater: IUpdater = Adam.Builder().learningRate(0.0002).beta1(0.5).build()): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder().apply {
        seed(seed)
        updater(updater)
        gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
        gradientNormalizationThreshold(100.0)
        weightInit(WeightInit.XAVIER)
        activation(IDENTITY)
    }.list().apply {
        for (layer in getDiscriminatorLayers(seed, updater)) layer(layer)
        inputType = InputType.convolutional(imageSize.toLong(), imageSize.toLong(), 3)
    }.build()

}

fun getGeneratorLayers(seed: Long = 123): Array<Layer> {
    return arrayOf(
        deconvLayer(
            32, 5, 2, 0, SELU, convolutionMode = ConvolutionMode.Same,
            channels = 64
        ),
        batchLayer(),

        deconvLayer(16, 5, 2, 0, SELU, convolutionMode = ConvolutionMode.Same),
        batchLayer(),

        deconvLayer(8, 5, 2, 0, SELU, convolutionMode = ConvolutionMode.Same),
        batchLayer(),
        deconvLayer(3, 5, 1, 0, TANH, convolutionMode = ConvolutionMode.Same)
    )
}

fun getDiscriminatorLayers(seed: Long = 123, updater: IUpdater = Sgd.builder().learningRate(0.0).build()): Array<Layer> {

    return arrayOf(
        convLayer(8, 5, 2, activation = LEAKYRELU, channels = 3, convolutionMode = ConvolutionMode.Same, updater=updater),

        convLayer(16, 5, 2, activation = LEAKYRELU, convolutionMode = ConvolutionMode.Same, updater=updater),

        convLayer(32, 5, 2, activation = LEAKYRELU, convolutionMode = ConvolutionMode.Same, updater=updater),

        convLayer(64, 5, 2, activation = LEAKYRELU, convolutionMode = ConvolutionMode.Same, updater=updater),

        convLayer(128, 5, 2, activation = LEAKYRELU, convolutionMode = ConvolutionMode.Same, updater=updater),

        convLayer(256, 5, 2, activation = LEAKYRELU, convolutionMode = ConvolutionMode.Same, updater=updater),

        // dropOutLayer(0.05, updater=updater),

        outputLayer(1, SIGMOID, XENT, updater=updater),
    )
}

fun copyParams(gen: MultiLayerNetwork, dis: MultiLayerNetwork, gan: MultiLayerNetwork) {
    gan.layers.forEachIndexed { index, layer ->
        if (index < gen.layers.size) gen.getLayer(index).setParams(layer.params())
        else dis.getLayer(index - gen.layers.size).setParams(layer.params())
    }
}

fun updateGan(gen: MultiLayerNetwork, dis: MultiLayerNetwork, gan: MultiLayerNetwork) {
    gan.layers.forEachIndexed { index, layer ->
        if (index >= gen.layers.size)
            layer.setParams(dis.getLayer(index - gen.layers.size).params())
    }
}