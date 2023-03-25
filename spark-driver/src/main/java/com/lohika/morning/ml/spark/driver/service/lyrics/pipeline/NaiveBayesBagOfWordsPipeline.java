package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
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
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.springframework.stereotype.Component;

@Component("NaiveBayesBagOfWordsPipeline")
public class NaiveBayesBagOfWordsPipeline extends CommonLyricsPipeline {

    public CrossValidatorModel classify() {
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

        CountVectorizer countVectorizer = new CountVectorizer().setInputCol(VERSE.getName()).setOutputCol("features");

        NaiveBayes naiveBayes = new NaiveBayes().setLabelCol(LABEL.getName());

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
                        countVectorizer,
                        naiveBayes});

        // Use a ParamGridBuilder to construct a grid of parameters to search over.
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{4, 8, 16})
                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(LABEL.getName()))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10);

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel model = crossValidator.fit(sentences);
        saveModel(model, getModelDirectory());

        return model;
    }

    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        modelStatistics.put("Sentences in verse", ((Verser) stages[8]).getSentencesInVerse());
        modelStatistics.put("Vocabulary", ((CountVectorizerModel) stages[9]).vocabulary().length);

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/naive-bayes/";
    }

}
