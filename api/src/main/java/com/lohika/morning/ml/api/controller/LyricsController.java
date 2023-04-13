package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.api.service.LyricsService;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins= {"*"}, maxAge = 4800, allowCredentials = "false" )
@RestController
@RequestMapping("/lyrics")
public class LyricsController {

    @Autowired
    private LyricsService lyricsService;

    @RequestMapping(value = "/train", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> trainLyricsModel() {
        Map<String, Object> trainStatistics = lyricsService.classifyLyrics();

        return new ResponseEntity<>(trainStatistics, HttpStatus.OK);
    }

    @RequestMapping(value = "/statistics", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> statisticsLyricsModel() {
        Map<String, Object> trainStatistics = lyricsService.getModelStatistics();

        return new ResponseEntity<>(trainStatistics, HttpStatus.OK);
    }

    @RequestMapping(value = "/predict", method = RequestMethod.POST)
    ResponseEntity<GenrePrediction> predictGenre(@RequestBody String unknownLyrics) {
        GenrePrediction genrePrediction = lyricsService.predictGenre(unknownLyrics);

        return new ResponseEntity<>(genrePrediction, HttpStatus.OK);
    }

}
