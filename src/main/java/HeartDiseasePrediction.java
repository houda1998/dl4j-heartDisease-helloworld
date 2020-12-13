import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.io.IOException;

public class HeartDiseasePrediction {
    public static void main(String[] args) throws IOException {
        String[] labels = {"ill", "healthy"};
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(new File("HeartDiseaseModel.zip"));
        System.out.println("Prédiction :");
        INDArray inputData= Nd4j.create(new double[][]{
                {63,1,3,145,233,1,0,150,0,2.3,0,0,1},
                {44,1,2,130,233,0,1,179,1,0.4,2,0,2},
                {46,1,0,140,311,0,1,120,1,1.8,1,2,3},
                {59,1,3,134,204,0,1,162,0,0.8,2,2,2},
                {57,1,1,154,232,0,0,164,0,0,2,1,2}
        });
        INDArray output=model.output(inputData);
        int[] classes=output.argMax(1).toIntVector();
        System.out.println(output);
        for(int i=0;i<classes.length;i++){
            System.out.println("Classe " +labels[classes[i]]);
        }
    }
}
