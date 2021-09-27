
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import javax.swing.JFrame
import javax.swing.JPanel

fun main() {
    networkTest()
}

fun networkTest() {
    val visualizationInterval = 1
    val new = true

    val gen: MultiLayerNetwork
    val dis: MultiLayerNetwork
    val gan: MultiLayerNetwork

    if (new) {
        gen = MultiLayerNetwork(getGenerator())
        dis = MultiLayerNetwork(getDiscriminator())
        gan = MultiLayerNetwork(getCombinedNetwork())
    } else {
        gen = ModelSerializer.restoreMultiLayerNetwork(File("gen.zip"), true)
        dis = ModelSerializer.restoreMultiLayerNetwork(File("dis.zip"), true)
        gan = ModelSerializer.restoreMultiLayerNetwork(File("gan.zip"), true)
    }

    gen.init()
    dis.init()
    gan.init()

    copyParams(gen, dis, gan)

    gen.setListeners(PerformanceListener(10, true))
    dis.setListeners(PerformanceListener(10, true))
    gan.setListeners(PerformanceListener(10, true))

    val frame: JFrame = GANVisualizationUtils.initFrame()
    val numSamples = 10
    val panel: JPanel = GANVisualizationUtils.initPanel(frame, numSamples)

    val trainData = getDataSetIterator()

    var j = 0
    while (true) {
        j++
        val real = trainData.next().features
        val batchSize = real.shape()[0]
        println(real.shapeInfoToString())

        val fakeIn = Nd4j.randn(batchSize, 7 * 7 * 64)
        val fake = gen.output(fakeIn)

        val realSet = DataSet(real, Nd4j.ones(batchSize, 1))
        val fakeSet = DataSet(fake, Nd4j.zeros(batchSize, 1))

        val data = DataSet.merge(listOf(realSet, fakeSet))

        repeat(2) { dis.fit(data) }

        updateGan(gen, dis, gan)

        gan.fit(DataSet(Nd4j.randn(batchSize, 7 * 7 * 64), Nd4j.zeros(batchSize, 1)))

        copyParams(gen, dis, gan)
        println("Iteration $j Visualizing...")
        val samples = arrayOfNulls<INDArray>(numSamples)
        val fakeSet2 = DataSet(fakeIn, Nd4j.ones(batchSize, 1))
        for (k in 0 until numSamples) {
            val input = fakeSet2[k].features
            samples[k] = gen.output(input)
        }
        GANVisualizationUtils.visualize(samples, frame, panel)

        ModelSerializer.writeModel(gen, "gen.zip", true)
        ModelSerializer.writeModel(dis, "dis.zip", true)
        ModelSerializer.writeModel(gan, "gan.zip", true)
        trainData.reset()
    }
}
