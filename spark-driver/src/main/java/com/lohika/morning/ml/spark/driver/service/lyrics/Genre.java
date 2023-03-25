package com.lohika.morning.ml.spark.driver.service.lyrics;

public enum Genre {


    POP("POP", 0D),
    COUNTRY("COUNTRY", 1D),
    BLUES("BLUES", 2D),
    JAZZ("JAZZ", 3D),
    REGGAE("REGGAE", 4D),
    ROCK("ROCK", 5D),
    HIPHOP("HIP HOP", 6D),

    UNKNOWN("Don\'t know :(", -1D);

    private final String name;
    private final Double value;

    Genre(final String name, final Double value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public Double getValue() {
        return value;
    }

}
