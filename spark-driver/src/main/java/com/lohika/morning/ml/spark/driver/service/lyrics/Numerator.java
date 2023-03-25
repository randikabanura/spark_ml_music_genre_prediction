package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class Numerator extends Transformer implements MLWritable {

    private String id = "id";
    private String rowNumber = "rowNumber";
    private String uid;

    public Numerator(String uid) {
        this.uid = uid;
    }

    public Numerator() {
        this.uid = "Numerator" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> sentences) {
        // Add unique id to each sentence of lyrics.
        Dataset<Row> sentencesWithIds = sentences.withColumn(id, functions.monotonically_increasing_id());
        sentencesWithIds = sentencesWithIds.withColumn(rowNumber, functions.row_number().over(Window.orderBy("id")));

        return sentencesWithIds;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema
                .add(DataTypes.createStructField(id, DataTypes.LongType, false))
                .add(DataTypes.createStructField(rowNumber, DataTypes.LongType, false));
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<Numerator> read() {
        return new DefaultParamsReader<>();
    }

}
