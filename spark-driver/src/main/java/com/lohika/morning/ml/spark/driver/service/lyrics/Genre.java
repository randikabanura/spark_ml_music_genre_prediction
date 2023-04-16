package com.lohika.morning.ml.spark.driver.service.lyrics;

public enum Genre {


    POP("POP", 0D), // 7042 records
    COUNTRY("COUNTRY", 1D), // 5445 records
    BLUES("BLUES", 2D), // 4604 records
    ROCK("ROCK", 3D), // 4034 records
    JAZZ("JAZZ", 4D), // 3839 records
    REGGAE("REGGAE", 5D), // 2498 records
    HIPHOP("HIPHOP", 6D), // 904 records
    METAL("METAL", 7D), // 4027 records

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
