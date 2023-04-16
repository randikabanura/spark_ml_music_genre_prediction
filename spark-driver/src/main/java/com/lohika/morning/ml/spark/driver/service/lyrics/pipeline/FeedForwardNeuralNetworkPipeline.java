package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import static org.apache.spark.sql.functions.rand;

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
        Dataset<Row> sentences = readLyrics();
        sentences = sentences.orderBy(rand());

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
                .setBlockSize(300)
                .setSeed(1234L)
                .setLayers(new int[]{300, 175, 70, 7});

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
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

        TrainValidationSplit trainSplitValidator = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                ;
                

        // Run cross-validation, and choose the best set of parameters.
        TrainValidationSplitModel model = trainSplitValidator.fit(sentences);

        saveModel(model, getModelDirectory());

        return model;
    }

    public Map<String, Object> getModelStatistics(TrainValidationSplitModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        modelStatistics.put("Sentences in verse", ((Verser) stages[7]).getSentencesInVerse());
        modelStatistics.put("Word2Vec vocabulary", ((Word2VecModel) stages[8]).getVectors().count());
        modelStatistics.put("Vector size", ((Word2VecModel) stages[8]).getVectorSize());
        modelStatistics.put("Weights", ((MultilayerPerceptronClassificationModel) stages[9]).weights());
        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    @Override
    public String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/feed-forward-neural-network/";
    }

}
