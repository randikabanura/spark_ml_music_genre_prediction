package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
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
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.ParamPair;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.springframework.stereotype.Component;

@Component("LogisticRegressionPipeline")
public class LogisticRegressionPipeline extends CommonLyricsPipeline {

    public TrainValidationSplitModel classify() {
        Dataset<Row> sentences = readLyrics();

        StringIndexer stringIndexer = new StringIndexer()
                .setInputCol(LABEL_STRING.getName())
                .setOutputCol(LABEL.getName());

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
                .setElasticNetParam(1D)
                .setLabelCol(LABEL.getName());

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
                        logisticRegression});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(word2Vec.stepSize(), new double[]{0.25, 0.5, 0.75, 0.9})
                .addGrid(verser.sentencesInVerse(), new int[]{4, 8, 12, 16})
                .addGrid(word2Vec.vectorSize(), new int[] {100, 200, 300})
                .addGrid(logisticRegression.regParam(), new double[] {0.1D, 0.01D, 0.001D, 0.0001D})
                .addGrid(logisticRegression.maxIter(), new int[] {100, 200})
                .build();

        Dataset<Row>[] splits = sentences.randomSplit(new double[] {0.9, 0.1}, 12345);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];


        TrainValidationSplit TrainValidationSplit = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEstimatorParamMaps(paramGrid)
                .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(LABEL.getName()).setMetricName("accuracy"))
                .setTrainRatio(0.8);

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
        modelStatistics.put("Reg parameter", ((LogisticRegressionModel) stages[10]).getRegParam());
        modelStatistics.put("Max iterations", ((LogisticRegressionModel) stages[10]).getMaxIter());

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/logistic-regression/";
    }

}
