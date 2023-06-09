package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import static org.apache.spark.sql.functions.rand;

import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;
import java.util.Map;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.NaiveBayes;
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

@Component("NaiveBayesTFIDFPipeline")
public class NaiveBayesTFIDFPipeline extends CommonLyricsPipeline {

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

        HashingTF tf = new HashingTF()
                .setInputCol(VERSE.getName())
                .setOutputCol("rawFeatures");

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");

        NaiveBayes naiveBayes = new NaiveBayes().setFeaturesCol("features");

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
                        tf,
                        idf,
                        naiveBayes});

        // Use a ParamGridBuilder to construct a grid of parameters to search over.
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{4, 8, 16, 32})
                .addGrid(tf.numFeatures(), new int[]{4096, 8192})
                .addGrid(idf.minDocFreq(), new int[]{0, 1, 2})
                .build();

        TrainValidationSplit trainSplitValidator = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid);
                

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
        modelStatistics.put("Num features", ((HashingTF) stages[8]).getNumFeatures());
        modelStatistics.put("Min doc frequency", ((IDFModel) stages[9]).getMinDocFreq());

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    @Override
    public String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/naive-bayes-tfidf/";
    }

}
