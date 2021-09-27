package school

import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.autodiff.listeners.impl.ScoreListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIteratorFactory
import org.nd4j.linalg.factory.Nd4j
import javax.swing.JFrame
import javax.swing.JPanel

fun main() {
    val gen = MultiLayerNetwork(getGenerator())
    val dis = MultiLayerNetwork(getDiscriminator())
    val gan = MultiLayerNetwork(getGan())

    gen.init()
    dis.init()
    gan.init()

    val frame: JFrame = GANVisualizationUtils.initFrame()
    val numSamples = 12
    val panel: JPanel = GANVisualizationUtils.initPanel(frame, numSamples)

    dis.setListeners(PerformanceListener(1, true))
    gan.setListeners(PerformanceListener(1, true))

    val batchSize = 100
    val dataSetList = mutableListOf<INDArray>()

    // creating the dataset
    repeat(20) {
        val batch = mutableListOf<INDArray>()
        repeat(batchSize) { batch.add(generateRGBImage(112)) }
        dataSetList.add(getBatchDataSetFromTensors(batch))
    }
    val dataSet = dataSetList.toList()

    var j = 0
    // GANVisualizationUtils.visualize(dataSet.toTypedArray(), frame, panel)
    while (true) {
        j++
        val real = dataSet.random()

        val fakeIn = Nd4j.randn(batchSize.toLong(), 64, 7, 7)
        val fake = gen.output(fakeIn)

        val realSet = DataSet(real, Nd4j.ones(batchSize, 1))
        val fakeSet = DataSet(fake, Nd4j.zeros(batchSize, 1))

        val data = DataSet.merge(listOf(realSet, fakeSet))

        dis.fit(data)

        updateGanFromDis(gen, dis, gan)

        gan.fit(DataSet(Nd4j.randn(batchSize.toLong(), 64, 7, 7), Nd4j.ones(batchSize, 1)))

        copyParamsToGan(gen, dis, gan)

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
    }
}