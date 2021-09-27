
import org.bytedeco.javacv.Java2DFrameUtils
import org.bytedeco.opencv.global.opencv_imgcodecs.imwrite
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.Size
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.math.roundToInt

const val imageSize = 112

fun main() {
    saveAugmentedDataset()
}

fun BufferedImage.toMat(): Mat = Java2DFrameUtils.toMat(this)

fun saveAugmentedDataset() {
    var index = 0
    File("real_images").listFiles().forEach {
        try {
            val image = (ImageIO.read(it))
            val mat = image.toMat()
            resize(mat, mat, Size(imageSize, imageSize))

            imwrite("augmented_images/$index.jpg", mat)
            index++
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}

fun getDataSetIterator(): RecordReaderDataSetIterator {
    val parentDir = File("augmented_images")
    val filesInDir = FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS)

    val recordReader = ImageRecordReader(imageSize.toLong(), imageSize.toLong(), 3)
    recordReader.initialize(filesInDir)
    return RecordReaderDataSetIterator(recordReader, 128)
}

fun getBufferedImagefromTensor(tensor: INDArray): BufferedImage {

    val image = BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB)

    for (i in 0 until imageSize) for (j in 0 until imageSize) {

        val red = (abs(tensor.getDouble(0, 0, i, j)) * 255).roundToInt()
        val green = (abs(tensor.getDouble(0, 1, i, j) * 255)).roundToInt()
        val blue = (abs(tensor.getDouble(0, 2, i, j) * 255)).roundToInt()
        val color = Color(red, green, blue).rgb

        image.setRGB(i, j, color)
    }

    return image
}