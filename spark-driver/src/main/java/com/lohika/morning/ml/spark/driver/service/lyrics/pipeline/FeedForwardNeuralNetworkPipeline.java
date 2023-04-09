package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;
import java.util.Map;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.springframework.stereotype.Component;

@Component("FeedForwardNeuralNetworkPipeline")
public class FeedForwardNeuralNetworkPipeline extends CommonLyricsPipeline {

    public TrainValidationSplitModel classify() {
        Dataset sentences = readLyrics();

        StringIndexer stringIndexer = new StringIndexer()
                .setInputCol(LABEL_STRING.getName())
                .setOutputCol(LABEL.getName());

        // Remove all punctuation symbols.
        Cleanser cleanser = new Cleanser();

        // Add rowNumber based on it.
        Numerator numerator = new Numerator();

        // Split into words.
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol(CLEAN.getName())
                .setOutputCol(WORDS.getName());

        // Remove stop words.
        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol(WORDS.getName())
                .setOutputCol(FILTERED_WORDS.getName());

        // Create as many rows as words. This is needed or Stemmer.
        Exploder exploder = new Exploder();

        // Perform stemming.
        Stemmer stemmer = new Stemmer();

        Uniter uniter = new Uniter();
        Verser verser = new Verser();

        // Create model.
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol(Column.VERSE.getName())
                .setOutputCol("features")
                .setMinCount(0);

        MultilayerPerceptronClassifier multilayerPerceptronClassifier = new MultilayerPerceptronClassifier()
                .setLabelCol(LABEL.getName())
                .setBlockSize(300)
                .setSeed(1234L)
                .setLayers(new int[]{300, 50, 2});

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
                        stringIndexer,
                        cleanser,
                        numerator,
                        tokenizer,
                        stopWordsRemover,
                        exploder,
                        stemmer,
                        uniter,
                        verser,
                        word2Vec,
                        multilayerPerceptronClassifier});

        // Use a ParamGridBuilder to construct a grid of parameters to search over.
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{16})
                .addGrid(word2Vec.vectorSize(), new int[] {300})
                .addGrid(multilayerPerceptronClassifier.maxIter(), new int[] {100})
                .build();

        Dataset<Row>[] splits = sentences.randomSplit(new double[] {0.8, 0.2}, 12345);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        TrainValidationSplit TrainValidationSplit = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(LABEL.getName()).setMetricName("accuracy"))
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.8);
                

        // Run cross-validation, and choose the best set of parameters.
        TrainValidationSplitModel model = TrainValidationSplit.fit(training);

        saveModel(model, getModelDirectory());

        model.transform(test)
                .select("features", LABEL.getName(), "prediction")
                .show();

        return model;
    }

    public Map<String, Object> getModelStatistics(TrainValidationSplitModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        modelStatistics.put("Sentences in verse", ((Verser) stages[8]).getSentencesInVerse());
        modelStatistics.put("Word2Vec vocabulary", ((Word2VecModel) stages[9]).getVectors().count());
        modelStatistics.put("Vector size", ((Word2VecModel) stages[9]).getVectorSize());
        modelStatistics.put("Weights", ((MultilayerPerceptronClassificationModel) stages[10]).weights());
        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/feed-forward-neural-network/";
    }

}
