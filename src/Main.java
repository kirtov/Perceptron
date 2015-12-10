import java.io.*;
import java.util.ArrayList;

/**
 * Created by dn1k on 05.12.2015.
 */
public class Main {
    public static void main(String[] args) throws IOException {
        MNISTReader reader = new MNISTReader();
        ArrayList<DigitImage> train = reader.loadDigitImages("train_labels.gz", "train_images.gz");
        ArrayList<DigitImage> test = reader.loadDigitImages("test_labels.gz", "test_images.gz");
        System.out.println(train.size());
        System.out.println(test.size());
//        Perceptron perc = new Perceptron(new int[]{784,500,150,10}, 0.005D);
//        perc.fit(train, false);
    }
}
