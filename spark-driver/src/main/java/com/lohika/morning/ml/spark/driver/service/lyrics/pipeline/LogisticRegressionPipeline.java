package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import static org.apache.spark.sql.functions.rand;

import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;
import java.util.Map;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.sql.functions.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.ParamPair;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.springframework.stereotype.Component;

@Component("LogisticRegressionPipeline")
public class LogisticRegressionPipeline extends CommonLyricsPipeline {

    public CrossValidatorModel classify() {
        Dataset<Row> sentences = readLyrics();
        sentences = sentences.orderBy(rand());

        Cleanser cleanser = new Cleanser();

        Numerator numerator = new Numerator();

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol(CLEAN.getName())
                .setOutputCol(WORDS.getName());

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol(WORDS.getName())
                .setOutputCol(FILTERED_WORDS.getName());

        Exploder exploder = new Exploder();

        Stemmer stemmer = new Stemmer();

        Uniter uniter = new Uniter();
        Verser verser = new Verser();

        Word2Vec word2Vec = new Word2Vec()
                                    .setInputCol(VERSE.getName())
                                    .setOutputCol("features")
                                    .setMinCount(0);

        LogisticRegression logisticRegression = new LogisticRegression()
                .setLabelCol(LABEL.getName());

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
                        logisticRegression});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{4, 8, 16})
                .addGrid(word2Vec.vectorSize(), new int[] {100, 200, 300})
                .addGrid(logisticRegression.regParam(), new double[] {0.01D})
                .addGrid(logisticRegression.maxIter(), new int[] {100, 200})
                .build();

        Dataset<Row>[] splits = sentences.randomSplit(new double[] {0.8, 0.2}, 3123);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3)
                .setEvaluator(new MulticlassClassificationEvaluator()
                        .setLabelCol(LABEL.getName()).setMetricName("accuracy"));

        CrossValidatorModel model = crossValidator.fit(training);

        saveModel(model, getModelDirectory());

        model.transform(test)
                .select("features", LABEL.getName(), "prediction")
                .show();

        return model;
    }

    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        modelStatistics.put("Sentences in verse", ((Verser) stages[7]).getSentencesInVerse());
        modelStatistics.put("Word2Vec vocabulary", ((Word2VecModel) stages[8]).getVectors().count());
        modelStatistics.put("Vector size", ((Word2VecModel) stages[8]).getVectorSize());
        modelStatistics.put("Reg parameter", ((LogisticRegressionModel) stages[9]).getRegParam());
        modelStatistics.put("Max iterations", ((LogisticRegressionModel) stages[9]).getMaxIter());

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/logistic-regression/";
    }

}
