package com.lohika.morning.ml.spark.distributed.library.function.map.generic.mllib;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;

public class MapRowToMLlibLabeledPoint implements Function<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) {
        return new LabeledPoint(inputRow.getAs(Column.LABEL.getName()), (DenseVector) inputRow.getAs("features"));
    }

}