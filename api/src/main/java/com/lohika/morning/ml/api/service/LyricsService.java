package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.MLService;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import com.lohika.morning.ml.spark.driver.service.lyrics.pipeline.LyricsPipeline;
import java.util.Map;
import javax.annotation.Resource;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class LyricsService {

    @Resource(name = "${lyrics.pipeline}")
    private LyricsPipeline pipeline;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    @Autowired
    private MLService mlService;

    public Map<String, Object> classifyLyrics() {
        TrainValidationSplitModel model = pipeline.classify();
        return pipeline.getModelStatistics(model);
    }

    public Map<String, Object> getModelStatistics() {
        TrainValidationSplitModel model = mlService.loadTrainValidationSplitModel(pipeline.getModelDirectory());
        return pipeline.getModelStatistics(model);
    }

    public GenrePrediction predictGenre(final String unknownLyrics) {
        return pipeline.predict(unknownLyrics);
    }

}
