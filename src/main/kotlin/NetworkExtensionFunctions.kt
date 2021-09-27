
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.*
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.IUpdater
import org.nd4j.linalg.lossfunctions.LossFunctions

fun convLayer(
    filters: Int,
    kernelSize: Int = 3,
    stride: Int = 1,
    padding: Int? = null,
    activation: Activation = Activation.LEAKYRELU,
    updater: IUpdater? = null,
    channels: Int? = null,
    dilation: Int? = null,
    convolutionMode: ConvolutionMode? = null
): ConvolutionLayer = ConvolutionLayer.Builder().apply {
    kernelSize(kernelSize, kernelSize)
    stride(stride, stride)
    if (padding != null) padding(padding, padding)
    activation(activation)
   if (updater != null) this.updater(updater)
    nOut(filters)
    if (channels != null) nIn(channels)
    if (dilation != null) dilation(dilation)
    if (convolutionMode != null) convolutionMode(convolutionMode)
}.build()

fun deconvLayer(
    filters: Int,
    kernelSize: Int = 3,
    stride: Int = 1,
    padding: Int = 1,
    activation: Activation = Activation.LEAKYRELU,
    updater: IUpdater? = null,
    channels: Int? = null,
    dilation: Int? = null,
    convolutionMode: ConvolutionMode? = null
): Deconvolution2D = Deconvolution2D.Builder().apply {
    kernelSize(kernelSize, kernelSize)
    stride(stride, stride)
    padding(padding, padding)
    activation(activation)
   if (updater != null) this.updater(updater)
    if (updater != null) this.updater(updater)
    nOut(filters)
    if (channels != null) nIn(channels)
    if (dilation != null) dilation(dilation)
    if (convolutionMode != null) convolutionMode(convolutionMode)
}.build()

fun NeuralNetConfiguration.ListBuilder.upSamplingLayer(
    size: Int
) {
    layer(Upsampling2D.Builder().apply {
        size(size)
    }.build())
}

fun poolingLayer(
    poolingType: SubsamplingLayer.PoolingType = SubsamplingLayer.PoolingType.MAX,
    kernelSize: Int = 2,
    stride: Int = 2,
    padding: Int = 1,
    dilation: Int? = null
): SubsamplingLayer = SubsamplingLayer.Builder().apply {
    kernelSize(kernelSize, kernelSize)
    stride(stride, stride)
    padding(padding, padding)
    poolingType(poolingType)
    if (dilation != null) dilation(dilation, dilation)
}.build()

fun batchLayer(): BatchNormalization = BatchNormalization()

fun dropOutLayer(dropOut: Double, output: Int? = null, activation: Activation = Activation.LEAKYRELU, updater: IUpdater?): DropoutLayer =
    DropoutLayer.Builder().apply {
        dropOut(dropOut)
        activation(activation)
        if (updater != null) this.updater(updater)
        if (output != null) nOut(output)
    }.build()

fun denseLayer(output: Int, activation: Activation = Activation.LEAKYRELU,
    updater: IUpdater? = null, bias: Boolean = true,
    input: Int? = null): DenseLayer = DenseLayer.Builder().apply {
        nOut(output)
        if (input != null) nIn(input)
        activation(activation)
   if (updater != null) this.updater(updater)
        isHasBias = true
    }.build()

fun outputLayer(
    output: Int,
    activation: Activation = Activation.SOFTMAX,
    lossFunction: LossFunctions.LossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD,
    updater: IUpdater?
): OutputLayer =
    OutputLayer.Builder().apply {
        lossFunction(lossFunction)
        nOut(output)
        activation(activation)
        if (updater != null) this.updater(updater)
    }.build()