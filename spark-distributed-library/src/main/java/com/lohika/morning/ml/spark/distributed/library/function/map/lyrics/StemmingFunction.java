package com.lohika.morning.ml.spark.distributed.library.function.map.lyrics;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.tartarus.snowball.SnowballProgram;
import org.tartarus.snowball.ext.EnglishStemmer;

public class StemmingFunction implements MapFunction<Row, Row> {

    @Override
    public Row call(Row input) throws Exception {
        SnowballProgram stemmer = new EnglishStemmer();
        stemmer.setCurrent(input.getAs(Column.FILTERED_WORD.getName()));
        stemmer.stem();
        String stemmedWord = stemmer.getCurrent();

        return RowFactory.create(
                input.get(input.schema().fieldIndex(Column.ID.getName())),
                input.getInt(input.schema().fieldIndex(Column.ROW_NUMBER.getName())),
                input.getDouble(input.schema().fieldIndex(Column.LABEL.getName())),
                stemmedWord);
    }

}
