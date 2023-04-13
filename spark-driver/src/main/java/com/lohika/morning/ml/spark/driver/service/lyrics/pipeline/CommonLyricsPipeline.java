package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import static org.apache.spark.sql.functions.*;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import com.lohika.morning.ml.spark.driver.service.MLService;
import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;

import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

public abstract class CommonLyricsPipeline implements LyricsPipeline {

    @Autowired
    protected SparkSession sparkSession;

    @Autowired
    private MLService mlService;

    @Value("${lyrics.training.set.directory.path}")
    private String lyricsTrainingSetDirectoryPath;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    @Override
    public GenrePrediction predict(final String unknownLyrics) {
        String lyrics[] = unknownLyrics.split("\\r?\\n");
        Dataset<String> lyricsDataset = sparkSession.createDataset(Arrays.asList(lyrics),
                Encoders.STRING());

        Dataset<Row> unknownLyricsDataset = lyricsDataset
                .withColumn(LABEL.getName(), functions.lit(Genre.UNKNOWN.getValue()))
                .withColumn(ID.getName(), functions.lit("unknown.txt"));

        unknownLyricsDataset = unknownLyricsDataset.withColumnRenamed("value", VALUE.getName());

        CrossValidatorModel model = mlService.loadCrossValidationModel(getModelDirectory());
        getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();

        Dataset<Row> predictionsDataset = bestModel.transform(unknownLyricsDataset);
        Row predictionRow = predictionsDataset.first();

        System.out.println("\n------------------------------------------------");
        final Double prediction = predictionRow.getAs("prediction");
        System.out.println("Prediction: " + Double.toString(prediction));

        if (Arrays.asList(predictionsDataset.columns()).contains("probability")) {
            final DenseVector probability = predictionRow.getAs("probability");
            System.out.println("Probability: " + probability);
            System.out.println("------------------------------------------------\n");

            return new GenrePrediction(getGenre(prediction).getName(),
                    probability.apply(Genre.POP.getValue().intValue()),
                    probability.apply(Genre.COUNTRY.getValue().intValue()),
                    probability.apply(Genre.BLUES.getValue().intValue()),
                    probability.apply(Genre.ROCK.getValue().intValue()),
                    probability.apply(Genre.JAZZ.getValue().intValue()),
                    probability.apply(Genre.REGGAE.getValue().intValue()),
                    probability.apply(Genre.HIPHOP.getValue().intValue()));
        }

        System.out.println("------------------------------------------------\n");
        return new GenrePrediction(getGenre(prediction).getName());
    }

    Dataset<Row> readLyrics() {
        Dataset input = readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.POP)
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.COUNTRY))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.BLUES))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.ROCK))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.JAZZ))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.REGGAE))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.HIPHOP));
        input = input.withColumnRenamed(input.columns()[0], ID.getName());
        input = input.withColumnRenamed(LABEL.getName(), LABEL_STRING.getName());
        // Reduce the input amount of partition minimal amount (spark.default.parallelism OR 2, whatever is less)
        input = input.coalesce(sparkSession.sparkContext().defaultMinPartitions()).cache();
        // Force caching.
        input.count();

        return input;
    }

    private Dataset<Row> readLyricsForGenre(String inputDirectory, Genre genre) {
        Dataset<Row> lyrics = readLyrics(inputDirectory);
        Dataset<Row> labeledLyrics = lyrics.filter(lyrics.col(LABEL.getName()).equalTo(genre.getName().toLowerCase()));
        System.out.println(genre.name() + " music sentences = " + labeledLyrics.count());

        return labeledLyrics;
    }

    private Dataset<Row> readLyrics(String inputDirectory) {
        Dataset<Row> rawLyrics = sparkSession.read().option("header", "true").csv(inputDirectory);
        rawLyrics = rawLyrics.filter(rawLyrics.col(VALUE.getName()).notEqual(""));
        rawLyrics = rawLyrics.filter(rawLyrics.col(VALUE.getName()).contains(" "));

        return rawLyrics;
    }

    private Genre getGenre(Double value) {
        for (Genre genre : Genre.values()) {
            if (genre.getValue().equals(value)) {
                return genre;
            }
        }

        return Genre.UNKNOWN;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

       /* Arrays.sort(model.);
        modelStatistics.put("Best model metrics", model.avgMetrics()[model.avgMetrics().length - 1]);*/

        return modelStatistics;
    }

    void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------------------------------------");
        System.out.println("Model statistics:");
        System.out.println(modelStatistics);
        System.out.println("------------------------------------------------\n");
    }

    void saveModel(TrainValidationSplit model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    void saveModel(CrossValidatorModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    void saveModel(PipelineModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    public void setLyricsTrainingSetDirectoryPath(String lyricsTrainingSetDirectoryPath) {
        this.lyricsTrainingSetDirectoryPath = lyricsTrainingSetDirectoryPath;
    }

    public void setLyricsModelDirectoryPath(String lyricsModelDirectoryPath) {
        this.lyricsModelDirectoryPath = lyricsModelDirectoryPath;
    }

    protected abstract String getModelDirectory();

    String getLyricsModelDirectoryPath() {
        return lyricsModelDirectoryPath;
    }
}
